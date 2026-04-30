// scripts/deploy.js
// Run with: npx hardhat run scripts/deploy.js --network sepolia
//
// Prerequisites:
//   npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
//   Set SEPOLIA_RPC_URL and PRIVATE_KEY in .env
//   npx hardhat compile

const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);
  console.log("Balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)), "ETH\n");

  // 1. Deploy ReputationOracle
  const Oracle = await ethers.getContractFactory("ReputationOracle");
  const oracle = await Oracle.deploy();
  await oracle.waitForDeployment();
  console.log("ReputationOracle:", await oracle.getAddress());

  // 2. Deploy LendingPool (depends on oracle)
  const Pool = await ethers.getContractFactory("LendingPool");
  const pool = await Pool.deploy(await oracle.getAddress());
  await pool.waitForDeployment();
  console.log("LendingPool:     ", await pool.getAddress());

  // 3. Deploy ScoreUpdater (depends on oracle; scorer = deployer for demo)
  const Updater = await ethers.getContractFactory("ScoreUpdater");
  const updater = await Updater.deploy(await oracle.getAddress(), deployer.address);
  await updater.waitForDeployment();
  console.log("ScoreUpdater:    ", await updater.getAddress());

  // 4. Wire up: set LendingPool on oracle, set ScoreUpdater as oracle owner
  console.log("\nConfiguring permissions...");
  await oracle.setLendingPool(await pool.getAddress());
  await oracle.transferOwnership(await updater.getAddress());
  console.log("  oracle.lendingPool → LendingPool ✓");
  console.log("  oracle.owner       → ScoreUpdater ✓");

  console.log("\n✅ Deployment complete. Save these addresses:\n");
  console.log(JSON.stringify({
    oracle:       await oracle.getAddress(),
    lendingPool:  await pool.getAddress(),
    scoreUpdater: await updater.getAddress(),
    network:      "sepolia",
    deployer:     deployer.address,
  }, null, 2));
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
