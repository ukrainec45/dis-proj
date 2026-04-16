# Structural model of UAV navigation system
**Purpose:** Describes the components that navigation system consists of. 

```mermaid 
graph TD
    %% Main Container
    subgraph Subsystem ["Навігаційне забезпечення БПС"]
        Sensors["Сенсори і датчики"]
        DataFusion["Модуль злиття даних"]
        Positioning["Модуль визначення положення у просторі"]
        RoutePlanning["Модуль визначення оптимального маршруту"]
        RouteControl["Модуль контролю маршруту"]
    end

    %% External Modules
    MissionPlanning["Модуль планування місії"]
    CommandGen["Генератор команд керування"]
    FlightController["Контролер польоту"]

    %% Internal Connections
    Sensors --> DataFusion
    DataFusion --> Positioning
    DataFusion --> RoutePlanning
    Positioning --> RouteControl
    RouteControl --> RoutePlanning
    RoutePlanning --> RouteControl

    %% External Connections
    MissionPlanning --> RoutePlanning
    RoutePlanning --> MissionPlanning
    RouteControl --> CommandGen
    CommandGen --> FlightController
    
    %% Feedback Loop
    FlightController --> Positioning

    %% Styling to match the simple block look
    classDef default fill:#fff,stroke:#000,stroke-width:1px;
    style Subsystem fill:none,stroke:#000,stroke-width:1px;