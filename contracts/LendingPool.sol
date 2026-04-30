// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./ReputationOracle.sol";

/**
 * @title LendingPool
 * @notice Core lending contract. Borrowers deposit ETH as collateral and
 *         borrow a mock stablecoin (represented as raw uint256 for simplicity).
 *
 * Key mechanics:
 *   1. Collateral requirement is fetched from ReputationOracle per borrower.
 *   2. Anti-farming cap is enforced inside the oracle's getCollateralRatioBP().
 *   3. Liquidation threshold = collateral_ratio × 0.95 (5% liquidation buffer).
 *   4. Score decay is passive — the oracle returns a downgraded tier if stale.
 *   5. After repayment, oracle.recordLoan() is called to update max loan size.
 *
 * This is a Sepolia testnet contract — ETH is the only collateral asset,
 * and "borrowed amount" is tracked as an internal accounting unit (USD-equivalent).
 * A real deployment would integrate Chainlink price feeds and ERC-20 tokens.
 */
contract LendingPool {

    // ── Structs ───────────────────────────────────────────────────────────────

    struct Position {
        uint256 collateralETH;      // ETH locked (in wei)
        uint256 borrowedUSD;        // Loan in USD-equivalent (18-decimal)
        uint256 collateralRatioBP;  // Ratio at time of borrow (basis points)
        uint256 openedAt;           // block.timestamp
    }

    // ── Storage ───────────────────────────────────────────────────────────────

    ReputationOracle public oracle;
    address          public owner;

    // Simulated ETH/USD price (18-decimal). Owner updates this; in production
    // use Chainlink AggregatorV3Interface.
    uint256 public ethPriceUSD = 2000e18;   // $2,000 default

    // 5% liquidation bonus paid to liquidator
    uint256 public constant LIQUIDATION_BONUS_BP = 500;

    // Minimum borrow: $50 USD-equivalent
    uint256 public constant MIN_BORROW_USD = 50e18;

    // Protocol fee on interest: 0.5% flat (simplified — no time-based interest)
    uint256 public constant PROTOCOL_FEE_BP = 50;

    mapping(address => Position) public positions;
    uint256 public totalBorrowedUSD;
    uint256 public totalCollateralETH;
    uint256 public accruedFeesUSD;

    // ── Events ────────────────────────────────────────────────────────────────

    event Borrowed(
        address indexed borrower,
        uint256 collateralETH,
        uint256 borrowedUSD,
        uint256 ratioBP
    );
    event Repaid(address indexed borrower, uint256 repaidUSD, uint256 collateralReturned);
    event Liquidated(
        address indexed borrower,
        address indexed liquidator,
        uint256 debtRepaid,
        uint256 collateralSeized
    );
    event PriceUpdated(uint256 newPrice);

    // ── Modifiers ─────────────────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "Pool: not owner");
        _;
    }

    modifier noActivePosition() {
        require(positions[msg.sender].borrowedUSD == 0, "Pool: position already open");
        _;
    }

    modifier hasActivePosition() {
        require(positions[msg.sender].borrowedUSD > 0, "Pool: no active position");
        _;
    }

    // ── Constructor ───────────────────────────────────────────────────────────

    constructor(address _oracle) {
        oracle = ReputationOracle(_oracle);
        owner  = msg.sender;
    }

    // ── Core: Borrow ──────────────────────────────────────────────────────────

    /**
     * @notice Open a borrowing position.
     * @param borrowUSD Requested loan in USD-equivalent (18-decimal).
     *
     * The function:
     *   1. Queries oracle for the caller's collateral ratio (with anti-farm cap).
     *   2. Computes required ETH collateral = (borrowUSD × ratioBP) / (10000 × ethPrice).
     *   3. Verifies msg.value covers required collateral.
     *   4. Records the position and updates oracle with new loan size.
     */
    function borrow(uint256 borrowUSD) external payable noActivePosition {
        require(borrowUSD >= MIN_BORROW_USD, "Pool: below minimum borrow");

        // Fetch personalised collateral ratio
        uint16  ratioBP = oracle.getCollateralRatioBP(msg.sender, borrowUSD);

        // Required collateral in ETH:
        //   requiredETH = (borrowUSD × ratioBP) / (10000 × ethPriceUSD)
        uint256 requiredCollateralUSD = (borrowUSD * ratioBP) / 10000;
        uint256 requiredCollateralETH = (requiredCollateralUSD * 1e18) / ethPriceUSD;

        require(msg.value >= requiredCollateralETH,
            "Pool: insufficient collateral sent");

        // Refund any excess ETH
        uint256 excessETH = msg.value - requiredCollateralETH;

        positions[msg.sender] = Position({
            collateralETH:     requiredCollateralETH,
            borrowedUSD:       borrowUSD,
            collateralRatioBP: ratioBP,
            openedAt:          block.timestamp
        });

        totalBorrowedUSD    += borrowUSD;
        totalCollateralETH  += requiredCollateralETH;

        // Notify oracle of loan size (for anti-farming cap tracking)
        oracle.recordLoan(msg.sender, borrowUSD);

        if (excessETH > 0) {
            (bool sent,) = msg.sender.call{value: excessETH}("");
            require(sent, "Pool: ETH refund failed");
        }

        emit Borrowed(msg.sender, requiredCollateralETH, borrowUSD, ratioBP);
    }

    // ── Core: Repay ───────────────────────────────────────────────────────────

    /**
     * @notice Repay the full loan and reclaim collateral.
     *         Protocol fee (0.5%) is deducted from returned collateral.
     *
     *         In a production system, repayment would be in a stablecoin token.
     *         Here, repayment is implicit — the borrower calls repay() and their
     *         debt is cleared. Fee is retained as protocol ETH.
     */
    function repay() external hasActivePosition {
        Position memory pos = positions[msg.sender];

        // Calculate protocol fee on the collateral
        uint256 feeETH = (pos.collateralETH * PROTOCOL_FEE_BP) / 10000;
        uint256 returnETH = pos.collateralETH - feeETH;

        // Accounting
        totalBorrowedUSD   -= pos.borrowedUSD;
        totalCollateralETH -= pos.collateralETH;
        accruedFeesUSD     += (feeETH * ethPriceUSD) / 1e18;

        delete positions[msg.sender];

        // Return collateral minus fee
        (bool sent,) = msg.sender.call{value: returnETH}("");
        require(sent, "Pool: ETH transfer failed");

        emit Repaid(msg.sender, pos.borrowedUSD, returnETH);
    }

    // ── Core: Liquidate ───────────────────────────────────────────────────────

    /**
     * @notice Liquidate an undercollateralised position.
     *
     * Health Factor = (collateralETH × ethPriceUSD × liqThresholdBP) / (borrowedUSD × 10000)
     * Liquidation condition: Health Factor < 1
     *   ↔ collateralETH × ethPriceUSD × liqThresholdBP < borrowedUSD × 10000
     *
     * Liquidation threshold = collateral_ratio × 0.95
     * Liquidator receives: seizedCollateral = debtETH × (1 + LIQUIDATION_BONUS_BP/10000)
     */
    function liquidate(address borrower) external {
        Position memory pos = positions[borrower];
        require(pos.borrowedUSD > 0, "Pool: no position");

        // Liquidation threshold = 95% of collateral ratio
        uint256 liqThresholdBP = (uint256(pos.collateralRatioBP) * 9500) / 10000;

        // Check health factor
        uint256 collateralValueUSD = (pos.collateralETH * ethPriceUSD) / 1e18;
        uint256 liqCollateralUSD   = (collateralValueUSD * liqThresholdBP) / 10000;

        require(liqCollateralUSD < pos.borrowedUSD, "Pool: position is healthy");

        // Compute collateral to seize: debt value + 5% bonus, capped at total collateral
        uint256 debtInETH     = (pos.borrowedUSD * 1e18) / ethPriceUSD;
        uint256 bonusETH      = (debtInETH * LIQUIDATION_BONUS_BP) / 10000;
        uint256 seizeETH      = _min(debtInETH + bonusETH, pos.collateralETH);

        // Bad debt (if any): shortfall absorbed by protocol reserve
        uint256 remainingCollateral = pos.collateralETH - seizeETH;

        totalBorrowedUSD   -= pos.borrowedUSD;
        totalCollateralETH -= pos.collateralETH;

        delete positions[borrower];

        // Any remaining collateral returned to borrower
        if (remainingCollateral > 0) {
            (bool sentBack,) = borrower.call{value: remainingCollateral}("");
            require(sentBack, "Pool: borrower refund failed");
        }

        // Send seized collateral to liquidator
        (bool sent,) = msg.sender.call{value: seizeETH}("");
        require(sent, "Pool: liquidator transfer failed");

        emit Liquidated(borrower, msg.sender, pos.borrowedUSD, seizeETH);
    }

    // ── View helpers ──────────────────────────────────────────────────────────

    /**
     * @notice Returns the current health factor of a position (18-decimal).
     *         Values < 1e18 indicate the position is liquidatable.
     */
    function healthFactor(address borrower) external view returns (uint256) {
        Position memory pos = positions[borrower];
        if (pos.borrowedUSD == 0) return type(uint256).max;

        uint256 liqThresholdBP   = (uint256(pos.collateralRatioBP) * 9500) / 10000;
        uint256 collateralUSD    = (pos.collateralETH * ethPriceUSD) / 1e18;
        uint256 thresholdUSD     = (collateralUSD * liqThresholdBP) / 10000;

        // HF = thresholdUSD / borrowedUSD, scaled to 18 decimals
        return (thresholdUSD * 1e18) / pos.borrowedUSD;
    }

    /**
     * @notice Preview: how much ETH collateral does a borrow require?
     */
    function previewCollateral(
        address borrower,
        uint256 borrowUSD
    ) external view returns (uint256 requiredETH, uint16 ratioBP) {
        ratioBP = oracle.getCollateralRatioBP(borrower, borrowUSD);
        uint256 collateralUSD = (borrowUSD * ratioBP) / 10000;
        requiredETH = (collateralUSD * 1e18) / ethPriceUSD;
    }

    /**
     * @notice Protocol solvency snapshot.
     */
    function solvencySnapshot() external view returns (
        uint256 totalBorrowed,
        uint256 totalCollateral,
        uint256 collateralValueUSD,
        uint256 coverageRatio
    ) {
        totalBorrowed     = totalBorrowedUSD;
        totalCollateral   = totalCollateralETH;
        collateralValueUSD = (totalCollateralETH * ethPriceUSD) / 1e18;
        coverageRatio     = totalBorrowedUSD > 0
            ? (collateralValueUSD * 1e18) / totalBorrowedUSD
            : type(uint256).max;
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    function updatePrice(uint256 newPrice) external onlyOwner {
        ethPriceUSD = newPrice;
        emit PriceUpdated(newPrice);
    }

    function withdrawFees() external onlyOwner {
        uint256 fees = address(this).balance - totalCollateralETH;
        require(fees > 0, "Pool: no fees");
        (bool sent,) = owner.call{value: fees}("");
        require(sent, "Pool: fee transfer failed");
    }

    receive() external payable {}

    // ── Internal ──────────────────────────────────────────────────────────────

    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
