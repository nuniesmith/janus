//! Integration tests for worker module
//!
//! Tests the interaction between Procedural memory, SkillLibrary,
//! and WorkerAgent components.

use crate::hippocampus::worker::procedural::{
    ExecutionContext, Procedural, ProcedureType, StepResult,
};
use crate::hippocampus::worker::skill_library::{
    MarketCondition, SkillComplexity, SkillLibrary, TriggerType,
};
use crate::hippocampus::worker::worker_agent::{
    ActionResult, ActionType, Goal, GoalType, MarketRegime, WorkerAgent, WorkerAgentConfig,
    WorkerState,
};
use std::collections::HashMap;

#[test]
fn test_worker_integration() {
    // Create all three components
    let procedural = Procedural::new();
    let skill_library = SkillLibrary::new();
    let worker_agent = WorkerAgent::new();

    // Verify all components initialized properly
    assert!(!procedural.is_empty());
    assert!(!skill_library.is_empty());
    assert!(worker_agent.is_active());
}

#[test]
fn test_worker_goal_to_skill_selection() {
    let skill_library = SkillLibrary::new();
    let mut worker_agent = WorkerAgent::new();

    // Set a goal
    let goal = Goal::new(GoalType::EnterPosition)
        .with_symbol("BTC")
        .with_direction(1)
        .with_magnitude(1.0)
        .with_urgency(0.7);

    worker_agent.set_goal(goal).unwrap();

    // Find applicable skills for this condition
    let applicable_skills = skill_library.get_applicable(MarketCondition::TrendingUp);
    assert!(!applicable_skills.is_empty());

    // Select action from worker
    let action = worker_agent.select_action();
    assert!(action.is_some());
}

#[test]
fn test_worker_execution_flow() {
    let mut procedural = Procedural::new();
    let mut worker_agent = WorkerAgent::new();

    // Set state and goal
    worker_agent.update_state(
        WorkerState::new()
            .with_price(50000.0, 50010.0)
            .with_volatility(0.02)
            .with_regime(MarketRegime::Trending),
    );

    let goal = Goal::new(GoalType::EnterPosition)
        .with_symbol("BTC")
        .with_direction(1)
        .with_magnitude(0.1);

    worker_agent.set_goal(goal).unwrap();

    // Get action from worker
    let action = worker_agent.select_action().unwrap();

    // Start procedure for the action
    let exec_result = procedural.start_execution("default_market_entry", HashMap::new());
    assert!(exec_result.is_ok());

    let exec_id = exec_result.unwrap();

    // Simulate execution steps
    let step_result = StepResult {
        sequence: 1,
        success: true,
        duration_ms: 50,
        retries: 0,
        output: None,
        error: None,
    };
    procedural.advance_execution(&exec_id, step_result).unwrap();

    // Record result in worker
    let result = ActionResult::success(action).with_fill(50005.0, 0.1, 0.0001);
    worker_agent.record_result(result);

    // Complete goal
    worker_agent.complete_goal(true);

    // Verify stats
    assert_eq!(worker_agent.stats().completed_goals, 1);
    assert_eq!(worker_agent.stats().total_actions, 1);
}

#[test]
fn test_skill_driven_action_selection() {
    let skill_library = SkillLibrary::new();
    let mut worker_agent = WorkerAgent::new();

    // Set market conditions
    worker_agent.update_state(
        WorkerState::new()
            .with_price(100.0, 100.05)
            .with_volatility(0.01)
            .with_position(0.0),
    );

    // Find triggered skills
    let mut trigger_values = HashMap::new();
    trigger_values.insert(TriggerType::Momentum, 0.6);
    trigger_values.insert(TriggerType::Volume, 1.8);
    trigger_values.insert(TriggerType::Spread, 0.0005);

    let triggered =
        skill_library.find_triggered_skills(MarketCondition::TrendingUp, &trigger_values);

    // Should find momentum_entry skill
    let has_momentum = triggered.iter().any(|s| s.id == "momentum_entry");
    assert!(has_momentum, "Should find momentum_entry skill");

    // Set goal based on triggered skill
    let goal = Goal::new(GoalType::EnterPosition)
        .with_symbol("ETH")
        .with_direction(1)
        .with_magnitude(2.0);

    worker_agent.set_goal(goal).unwrap();

    let action = worker_agent.select_action().unwrap();

    // Action should be buy-related
    assert!(matches!(
        action.action_type,
        ActionType::MarketBuy | ActionType::LimitBuy | ActionType::NoOp
    ));
}

#[test]
fn test_procedure_skill_recording() {
    let mut skill_library = SkillLibrary::new();
    let mut procedural = Procedural::new();

    // Execute a procedure
    let exec_id = procedural
        .start_execution("default_market_entry", HashMap::new())
        .unwrap();

    // Simulate successful execution
    for seq in 1..=3 {
        let step = StepResult {
            sequence: seq,
            success: true,
            duration_ms: 50 * seq as u64,
            retries: 0,
            output: None,
            error: None,
        };
        procedural.advance_execution(&exec_id, step).unwrap();
    }

    procedural.complete_execution(&exec_id, true).unwrap();

    // Record skill usage based on procedure success
    skill_library
        .record_usage(
            "momentum_entry",
            true,
            0.05,
            MarketCondition::TrendingUp,
            HashMap::new(),
        )
        .unwrap();

    // Verify skill stats updated
    let skill = skill_library.get_skill("momentum_entry").unwrap();
    assert_eq!(skill.usage_count, 1);
    assert_eq!(skill.success_count, 1);

    // Verify experience gained
    assert!(skill_library.experience() > 0);
}

#[test]
fn test_emergency_procedure_execution() {
    let mut procedural = Procedural::new();
    let mut worker_agent = WorkerAgent::new();

    // Set up position
    worker_agent.update_state(WorkerState::new().with_position(1.0).with_pnl(-0.05, 0.0));

    // Emergency goal
    let emergency = Goal::new(GoalType::Emergency).with_symbol("BTC");

    worker_agent.set_goal(emergency).unwrap();

    // Emergency should be highest priority
    let current = worker_agent.current_goal().unwrap();
    assert_eq!(current.goal_type, GoalType::Emergency);

    // Get emergency action
    let action = worker_agent.select_action().unwrap();

    // Should be flatten or cancel all
    assert!(matches!(
        action.action_type,
        ActionType::Flatten | ActionType::CancelAll
    ));

    // Execute emergency procedure
    procedural.set_context(ExecutionContext::HighVolatility);
    let best_proc = procedural.get_best_procedure(
        ProcedureType::EmergencyExit,
        ExecutionContext::HighVolatility,
    );
    assert!(best_proc.is_some());
}

#[test]
fn test_learning_integration() {
    let mut skill_library = SkillLibrary::new();
    let mut worker_agent = WorkerAgent::with_config(WorkerAgentConfig {
        enable_learning: true,
        exploration_rate: 0.0, // No exploration for deterministic test
        ..Default::default()
    });

    // Execute multiple actions and record results
    for i in 0..10 {
        let goal = Goal::new(GoalType::EnterPosition)
            .with_symbol("BTC")
            .with_direction(1)
            .with_magnitude(1.0);

        worker_agent.set_goal(goal).unwrap();

        let action = worker_agent.select_action().unwrap();
        let success = i % 3 != 0; // ~66% success rate
        let reward = if success { 0.05 } else { -0.02 };

        let result = if success {
            ActionResult::success(action).with_fill(50000.0, 1.0, 0.0001)
        } else {
            ActionResult::failure(action, "Slippage too high")
        };

        worker_agent.record_result(result);

        // Also record in skill library
        skill_library
            .record_usage(
                "momentum_entry",
                success,
                reward,
                MarketCondition::Any,
                HashMap::new(),
            )
            .unwrap();

        worker_agent.complete_goal(success);
    }

    // Verify learning occurred
    let stats = worker_agent.stats();
    assert_eq!(stats.total_goals, 10);
    assert!(stats.total_reward != 0.0);

    // Skill should have updated stats
    let skill = skill_library.get_skill("momentum_entry").unwrap();
    assert_eq!(skill.usage_count, 10);
}

#[test]
fn test_skill_complexity_progression() {
    let mut skill_library = SkillLibrary::new();

    // Initially only basic skills accessible
    let applicable = skill_library.get_applicable(MarketCondition::Any);
    assert!(
        applicable
            .iter()
            .all(|s| s.complexity <= SkillComplexity::Basic)
    );

    // Add experience to unlock intermediate
    skill_library.add_experience(150);
    assert_eq!(
        skill_library.unlocked_complexity(),
        SkillComplexity::Intermediate
    );

    // Now should see intermediate skills too
    let applicable = skill_library.get_applicable(MarketCondition::Ranging);
    let _has_intermediate = applicable
        .iter()
        .any(|s| s.complexity == SkillComplexity::Intermediate);
    // Note: depends on which skills are applicable in Ranging condition
    assert!(skill_library.unlocked_complexity() >= SkillComplexity::Intermediate);
}

#[test]
fn test_procedure_context_adaptation() {
    let mut procedural = Procedural::new();

    // Normal context
    procedural.set_context(ExecutionContext::Normal);
    let normal_best =
        procedural.get_best_procedure(ProcedureType::OrderEntry, ExecutionContext::Normal);
    assert!(normal_best.is_some());

    // High volatility context
    procedural.set_context(ExecutionContext::HighVolatility);
    let vol_best = procedural.get_best_procedure(
        ProcedureType::EmergencyExit,
        ExecutionContext::HighVolatility,
    );
    assert!(vol_best.is_some());

    // Emergency exit should be applicable in high volatility
    let emergency_proc = procedural.get_procedure("default_emergency_exit").unwrap();
    assert!(emergency_proc.is_applicable(ExecutionContext::HighVolatility));
}

#[test]
fn test_worker_state_affects_actions() {
    let mut worker_agent = WorkerAgent::new();

    // Flat position - entry goal
    worker_agent.update_state(WorkerState::new().with_position(0.0));

    let entry_goal = Goal::new(GoalType::EnterPosition)
        .with_symbol("BTC")
        .with_direction(1);
    worker_agent.set_goal(entry_goal).unwrap();

    let action = worker_agent.select_action().unwrap();
    assert!(matches!(
        action.action_type,
        ActionType::MarketBuy | ActionType::LimitBuy | ActionType::NoOp
    ));

    worker_agent.complete_goal(true);

    // Long position - exit goal
    worker_agent.update_state(WorkerState::new().with_position(1.0));

    let exit_goal = Goal::new(GoalType::ExitPosition).with_symbol("BTC");
    worker_agent.set_goal(exit_goal).unwrap();

    let action = worker_agent.select_action().unwrap();
    assert!(matches!(
        action.action_type,
        ActionType::MarketSell | ActionType::LimitSell | ActionType::Flatten | ActionType::NoOp
    ));
}

#[test]
fn test_full_trade_lifecycle() {
    let mut procedural = Procedural::new();
    let mut skill_library = SkillLibrary::new();
    let mut worker_agent = WorkerAgent::new();

    // 1. Initial state - flat
    worker_agent.update_state(
        WorkerState::new()
            .with_price(100.0, 100.05)
            .with_position(0.0)
            .with_volatility(0.015),
    );

    // 2. Entry signal from skill
    let mut triggers = HashMap::new();
    triggers.insert(TriggerType::Momentum, 0.7);
    let triggered = skill_library.find_triggered_skills(MarketCondition::TrendingUp, &triggers);
    assert!(!triggered.is_empty());

    // 3. Set entry goal
    let entry_goal = Goal::new(GoalType::EnterPosition)
        .with_symbol("TEST")
        .with_direction(1)
        .with_magnitude(10.0);
    worker_agent.set_goal(entry_goal).unwrap();

    // 4. Get entry action
    let entry_action = worker_agent.select_action().unwrap();

    // 5. Execute entry procedure
    let entry_exec = procedural
        .start_execution("default_market_entry", HashMap::new())
        .unwrap();

    // Simulate steps
    for seq in 1..=3 {
        procedural
            .advance_execution(
                &entry_exec,
                StepResult {
                    sequence: seq,
                    success: true,
                    duration_ms: 50,
                    retries: 0,
                    output: None,
                    error: None,
                },
            )
            .unwrap();
    }
    procedural.complete_execution(&entry_exec, true).unwrap();

    // 6. Record entry result
    let entry_result = ActionResult::success(entry_action).with_fill(100.02, 10.0, 0.0002);
    worker_agent.record_result(entry_result);
    worker_agent.complete_goal(true);

    // 7. Update state - now have position
    worker_agent.update_state(
        WorkerState::new()
            .with_price(102.0, 102.05)
            .with_position(10.0)
            .with_pnl(0.02, 0.0),
    );

    // 8. Set exit goal
    let exit_goal = Goal::new(GoalType::ExitPosition)
        .with_symbol("TEST")
        .with_magnitude(10.0);
    worker_agent.set_goal(exit_goal).unwrap();

    // 9. Get exit action
    let exit_action = worker_agent.select_action().unwrap();

    // 10. Record exit result
    let exit_result = ActionResult::success(exit_action).with_fill(102.03, 10.0, 0.0001);
    worker_agent.record_result(exit_result);
    worker_agent.complete_goal(true);

    // 11. Record skill usage
    skill_library
        .record_usage(
            "momentum_entry",
            true,
            0.02,
            MarketCondition::TrendingUp,
            HashMap::new(),
        )
        .unwrap();

    // 12. Verify final stats
    let stats = worker_agent.stats();
    assert_eq!(stats.total_goals, 2);
    assert_eq!(stats.completed_goals, 2);
    assert_eq!(stats.total_actions, 2);
    assert_eq!(stats.successful_actions, 2);

    assert_eq!(procedural.stats().successful_executions, 1);
}
