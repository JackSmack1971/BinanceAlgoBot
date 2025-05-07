# Architectural Documentation

This directory contains the architectural documentation for the project.

## Diagrams

*   [Component Diagram](component_diagram.plantuml): Shows the main components of the system and their relationships.
*   [Data Flow Diagram](data_flow_diagram.plantuml): Illustrates the flow of data during a backtest.
*   [Deployment Diagram](deployment_diagram.plantuml): Shows the deployment environment of the system.
*   [Sequence Diagrams](sequence_diagram_live_trading.plantuml):
    *   [Live Trading Cycle](sequence_diagram_live_trading.plantuml): Shows the sequence of events during a live trading cycle.
    *   [Backtest Execution](sequence_diagram_backtest_execution.plantuml): Shows the sequence of events during a backtest execution.
    *   [Strategy Switching Process](sequence_diagram_strategy_switching.plantuml): Shows the sequence of events during a strategy switching process.
*   [Entity-Relationship Diagram](erd.plantuml): Illustrates the database schema.

## Component Interface Specifications

*   [Strategy Interface](strategy_interface.md): Documents the `Strategy` interface.
*   [ExchangeInterface](exchange_interface.md): Documents the `ExchangeInterface`.
*   [PositionManager](position_manager_interface.md): Documents the `PositionManager`.
*   [RiskManagement](risk_management_interface.md): Documents the `RiskManagement`.

## Data Flow Documentation

*   [Data Flow](data_flow.md): Describes the data flow during a backtest.

## Database Schema Documentation

*   [Database Schema](database_schema.md): Documents the database schema.

## Configuration Management Documentation

*   [Configuration Management](configuration_management.md): Documents how configuration parameters are managed.

## Error Handling Strategy Documentation

*   [Error Handling](error_handling.md): Documents the error handling strategy.

## Viewing the Diagrams

The diagrams are created using PlantUML. To view the diagrams, you will need to install PlantUML and a PlantUML viewer.

For VS Code, you can use the PlantUML extension.

To generate images from the PlantUML files, you can use the following command:

```bash
plantuml *.plantuml