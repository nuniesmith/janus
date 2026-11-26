# AGENTS.md — janus

Rust ML inference engine and neuromorphic trading brain: a 39-crate workspace (ML inference, GPU compute, signal generation, gRPC services, QuestDB pipelines, risk management) that builds and runs standalone, and is also consumed as a service by the `fks` orchestrator.

## Shared context

Read the shared agent context first: `soul.md` and `memory.md` (especially §5) in the [agents repo](https://github.com/nuniesmith/agents).

## Standing facts

- **Janus's edge is NOT proven.** The Gate-A pre-registered test was not passed (44 trades; the 9-gate is a cost filter, not a signal — no gross separation). Never present Janus output as a proven signal, in code, docs or UI copy.
- Its live role in the platform is **read-only / advisory**: a daily logged regime assessment for the crypto bot (authenticated regime gRPC server, port 50052; aggregation = 60% directional time-share over the 00:05→00:05 UTC window). The bot binary does not take orders from Janus.
- Governance records (pre-registrations, verdicts) live in the private `fks-governance` repo — never in this public repo.

## Working here

- Verify with the workspace's own checks (`cargo test`, clippy with `-D warnings`) and the repo CI before calling anything done; a PR is ready only when its own CI is green on its own head.
- One Rithmic session per credential applies anywhere Rithmic is touched: a "permission denied" means another session is active, not a missing entitlement.
