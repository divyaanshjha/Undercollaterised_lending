// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ReputationOracle
 * @notice Stores a collateral tier for each borrower wallet, set by an
 *         off-chain scorer (the Python model) or, in production, a ZK verifier.
 *
 * Tiers map to collateral ratios (in basis points, where 10000 = 100%):
 *   TIER_NEW      → 15000  (150% — no history, Aave default)
 *   TIER_BRONZE   → 14000  (140%)
 *   TIER_SILVER   → 13000  (130%)
 *   TIER_GOLD     → 12000  (120% — minimum floor)
 *
 * Anti-farming: the oracle also stores each wallet's max historical loan.
 * The LendingPool uses this to cap the discount on any new loan.
 *
 * Access control: only the contract owner (multisig in production) can
 * push score updates. Borrowers cannot update their own score.
 */
contract ReputationOracle {

    // ── Types ────────────────────────────────────────────────────────────────

    enum Tier { NEW, BRONZE, SILVER, GOLD }

    struct WalletProfile {
        Tier    tier;
        uint256 maxHistoricalLoanUSD;   // 18-decimal fixed-point, USD value
        uint256 lastUpdated;            // block.timestamp of last score push
    }

    // ── Storage ──────────────────────────────────────────────────────────────

    address public owner;
    address public lendingPool;          // authorised to record new loans

    mapping(address => WalletProfile) private profiles;

    // Basis-point collateral ratios per tier
    uint16 public constant RATIO_NEW    = 15000;
    uint16 public constant RATIO_BRONZE = 14000;
    uint16 public constant RATIO_SILVER = 13000;
    uint16 public constant RATIO_GOLD   = 12000;

    // Score decay: if no update in 180 days, tier drops one level
    uint256 public constant DECAY_PERIOD = 180 days;

    // Anti-farming cap multiplier (2.5×, stored as basis points: 25000 / 10000)
    uint256 public constant ANTI_FARM_MULTIPLIER_BP = 25000;

    // ── Events ───────────────────────────────────────────────────────────────

    event TierUpdated(address indexed wallet, Tier oldTier, Tier newTier);
    event MaxLoanUpdated(address indexed wallet, uint256 newMax);
    event ScoreDecayed(address indexed wallet, Tier fromTier, Tier toTier);

    // ── Modifiers ────────────────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "Oracle: not owner");
        _;
    }

    modifier onlyLendingPool() {
        require(msg.sender == lendingPool, "Oracle: not lending pool");
        _;
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    constructor() {
        owner = msg.sender;
    }

    function setLendingPool(address _pool) external onlyOwner {
        lendingPool = _pool;
    }

    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "Oracle: zero address");
        owner = _newOwner;
    }

    // ── Score management (owner-only) ─────────────────────────────────────────

    /**
     * @notice Push a new tier for a wallet (called by off-chain scorer or ZK verifier).
     */
    function setTier(address wallet, Tier tier) external onlyOwner {
        Tier old = profiles[wallet].tier;
        profiles[wallet].tier        = tier;
        profiles[wallet].lastUpdated = block.timestamp;
        emit TierUpdated(wallet, old, tier);
    }

    /**
     * @notice Batch update — gas-efficient for pushing many wallet scores at once.
     */
    function setTierBatch(
        address[] calldata wallets,
        Tier[]    calldata tiers
    ) external onlyOwner {
        require(wallets.length == tiers.length, "Oracle: length mismatch");
        for (uint256 i = 0; i < wallets.length; i++) {
            Tier old = profiles[wallets[i]].tier;
            profiles[wallets[i]].tier        = tiers[i];
            profiles[wallets[i]].lastUpdated = block.timestamp;
            emit TierUpdated(wallets[i], old, tiers[i]);
        }
    }

    /**
     * @notice Called by LendingPool after a successful repayment to record
     *         the loan size for the anti-farming cap.
     */
    function recordLoan(address wallet, uint256 loanUSD) external onlyLendingPool {
        if (loanUSD > profiles[wallet].maxHistoricalLoanUSD) {
            profiles[wallet].maxHistoricalLoanUSD = loanUSD;
            emit MaxLoanUpdated(wallet, loanUSD);
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /**
     * @notice Returns the current tier of a wallet, applying decay if stale.
     */
    function getTier(address wallet) public view returns (Tier) {
        WalletProfile memory p = profiles[wallet];
        if (p.lastUpdated == 0) return Tier.NEW;   // never scored

        // Apply decay: if lastUpdated > DECAY_PERIOD ago, drop one tier
        if (block.timestamp > p.lastUpdated + DECAY_PERIOD) {
            if (p.tier == Tier.GOLD)   return Tier.SILVER;
            if (p.tier == Tier.SILVER) return Tier.BRONZE;
            // BRONZE and NEW decay to NEW
            return Tier.NEW;
        }
        return p.tier;
    }

    /**
     * @notice Returns the collateral ratio in basis points for a wallet,
     *         applying the anti-farming cap for the requested loan amount.
     *
     * @param wallet      Borrower address
     * @param loanUSD     Requested loan value in 18-decimal USD
     * @return ratioBP    Effective collateral ratio in basis points (e.g. 13000 = 130%)
     */
    function getCollateralRatioBP(
        address wallet,
        uint256 loanUSD
    ) external view returns (uint16 ratioBP) {
        Tier tier = getTier(wallet);
        uint16 tierRatio = _tierToRatio(tier);

        // Anti-farming cap: discount applies only up to 2.5× max historical loan
        uint256 maxDiscounted = (profiles[wallet].maxHistoricalLoanUSD *
                                 ANTI_FARM_MULTIPLIER_BP) / 10000;

        if (maxDiscounted == 0 || loanUSD <= maxDiscounted) {
            // Loan within historical range — full tier discount applies
            return tierRatio;
        }

        // Loan exceeds cap: blend tier rate for capped portion, 150% for excess
        // blended = tierRatio × (maxDiscounted/loanUSD) + 15000 × (excess/loanUSD)
        uint256 cappedFractionBP = (maxDiscounted * 10000) / loanUSD;
        uint256 excessFractionBP = 10000 - cappedFractionBP;
        uint256 blended = (uint256(tierRatio) * cappedFractionBP +
                           uint256(RATIO_NEW)  * excessFractionBP) / 10000;
        return uint16(blended);
    }

    /**
     * @notice Returns full profile for a wallet (for UI / analytics).
     */
    function getProfile(address wallet) external view returns (
        Tier    tier,
        uint16  ratioBP,
        uint256 maxHistoricalLoan,
        uint256 lastUpdated
    ) {
        tier              = getTier(wallet);
        ratioBP           = _tierToRatio(tier);
        maxHistoricalLoan = profiles[wallet].maxHistoricalLoanUSD;
        lastUpdated       = profiles[wallet].lastUpdated;
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    function _tierToRatio(Tier tier) internal pure returns (uint16) {
        if (tier == Tier.GOLD)   return RATIO_GOLD;
        if (tier == Tier.SILVER) return RATIO_SILVER;
        if (tier == Tier.BRONZE) return RATIO_BRONZE;
        return RATIO_NEW;
    }
}
