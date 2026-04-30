// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./ReputationOracle.sol";

/**
 * @title ScoreUpdater
 * @notice Receives signed score proofs from the off-chain model and pushes
 *         tier updates to the ReputationOracle.
 *
 * In production, this contract would verify a ZK proof (e.g. from Circom/snarkjs)
 * that the wallet's reputation score exceeds a threshold, without revealing the
 * underlying wallet history. For the academic prototype, it accepts signed messages
 * from a trusted off-chain scorer (the Python logistic regression model).
 *
 * Proof format (EIP-712 compatible):
 *   struct ScoreProof {
 *     address wallet;
 *     uint8   tier;       // 0=NEW, 1=BRONZE, 2=SILVER, 3=GOLD
 *     uint256 validUntil; // proof expiry timestamp
 *     uint256 nonce;      // replay protection
 *   }
 *
 * The off-chain scorer signs this struct; the wallet submits the signature
 * to claim their tier. This keeps scoring trustless — the oracle can't
 * arbitrarily upgrade a wallet; it must produce a valid scorer signature.
 */
contract ScoreUpdater {

    // ── EIP-712 domain ────────────────────────────────────────────────────────

    bytes32 public constant DOMAIN_TYPEHASH = keccak256(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    );

    bytes32 public constant PROOF_TYPEHASH = keccak256(
        "ScoreProof(address wallet,uint8 tier,uint256 validUntil,uint256 nonce)"
    );

    bytes32 public immutable DOMAIN_SEPARATOR;

    // ── Storage ───────────────────────────────────────────────────────────────

    ReputationOracle public oracle;
    address          public scorer;     // off-chain model signing key
    address          public owner;

    mapping(address => uint256) public nonces;   // replay protection per wallet

    // ── Events ────────────────────────────────────────────────────────────────

    event TierClaimed(address indexed wallet, ReputationOracle.Tier tier, uint256 nonce);
    event ScorerUpdated(address oldScorer, address newScorer);

    // ── Constructor ───────────────────────────────────────────────────────────

    constructor(address _oracle, address _scorer) {
        oracle = ReputationOracle(_oracle);
        scorer = _scorer;
        owner  = msg.sender;

        DOMAIN_SEPARATOR = keccak256(abi.encode(
            DOMAIN_TYPEHASH,
            keccak256("ReputationProtocol"),
            keccak256("1"),
            block.chainid,
            address(this)
        ));
    }

    // ── Tier claiming ─────────────────────────────────────────────────────────

    /**
     * @notice A wallet submits a signed proof from the off-chain scorer to
     *         claim their reputation tier on-chain.
     *
     * @param tier        Claimed tier (0–3)
     * @param validUntil  Proof expiry (unix timestamp)
     * @param v, r, s     ECDSA signature components
     *
     * Security properties:
     *   - Only valid scorer signatures accepted (unforgeable)
     *   - Proof expires (prevents stale proofs)
     *   - Nonce prevents replay attacks
     *   - Wallet cannot self-upgrade (scorer must sign based on actual on-chain history)
     */
    function claimTier(
        uint8   tier,
        uint256 validUntil,
        uint8   v,
        bytes32 r,
        bytes32 s
    ) external {
        require(block.timestamp <= validUntil, "Updater: proof expired");
        require(tier <= 3, "Updater: invalid tier");

        uint256 nonce = nonces[msg.sender]++;

        // Reconstruct EIP-712 digest
        bytes32 structHash = keccak256(abi.encode(
            PROOF_TYPEHASH,
            msg.sender,
            tier,
            validUntil,
            nonce
        ));
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash));

        // Recover signer and verify it's the trusted scorer
        address recovered = ecrecover(digest, v, r, s);
        require(recovered == scorer, "Updater: invalid signature");

        // Push tier to oracle
        oracle.setTier(msg.sender, ReputationOracle.Tier(tier));

        emit TierClaimed(msg.sender, ReputationOracle.Tier(tier), nonce);
    }

    /**
     * @notice Owner can directly push tiers (used for batch initialisation
     *         or emergency overrides — equivalent to oracle.setTierBatch).
     */
    function pushTierBatch(
        address[] calldata wallets,
        uint8[]   calldata tiers
    ) external {
        require(msg.sender == owner, "Updater: not owner");
        require(wallets.length == tiers.length, "Updater: length mismatch");

        ReputationOracle.Tier[] memory tierEnums = new ReputationOracle.Tier[](tiers.length);
        for (uint256 i = 0; i < tiers.length; i++) {
            require(tiers[i] <= 3, "Updater: invalid tier");
            tierEnums[i] = ReputationOracle.Tier(tiers[i]);
        }
        oracle.setTierBatch(wallets, tierEnums);
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    function updateScorer(address newScorer) external {
        require(msg.sender == owner, "Updater: not owner");
        require(newScorer != address(0), "Updater: zero address");
        emit ScorerUpdated(scorer, newScorer);
        scorer = newScorer;
    }
}
