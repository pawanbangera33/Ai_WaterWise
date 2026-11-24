# AI WaterWise Synthetic Dataset Documentation

## Overview
The `water_data.csv` dataset contains synthetic hourly records simulating an urban/institutional water distribution facility. The dataset includes environmental factors, building occupancy metrics, storage tank levels, and hourly water consumption.

- **Total Records**: 5,500 hourly observations (approx. 229 days)
- **Time Granularity**: 1-hour interval
- **Seed**: 42 (Reproducible generation)

## Features & Schema

| Column Name | Type | Unit / Range | Description |
|---|---|---|---|
| `timestamp` | Datetime String | `YYYY-MM-DD HH:MM:SS` | Hourly observation timestamp |
| `temperature` | Float | 12.0 - 42.0 °C | Ambient outdoor temperature |
| `humidity` | Float | 30.0 - 95.0 % | Relative atmospheric humidity |
| `occupancy` | Integer | 0 - 100 people | Number of active occupants in building/zone |
| `day_of_week` | Integer | 0 - 6 | Day index (0 = Monday, 6 = Sunday) |
| `is_holiday` | Integer | 0 or 1 | Binary indicator for public holidays |
| `tank_level` | Float | 5.0 - 98.0 % | Water storage overhead tank fill level |
| `water_consumption` | Float | Liters / Hour | Total observed hourly water consumption |

## Realistic Behavioral Patterns Encoded
1. **Diurnal Cycles**: Morning (06:00 - 09:00) and evening (18:00 - 21:00) peak consumption; late-night lull (01:00 - 05:00).
2. **Weather Correlation**: Higher temperatures drive increased HVAC/cooling and domestic water consumption.
3. **Occupancy Impact**: Consumption directly correlates with building population.
4. **Synthetic Abnormalities**:
   - High night flow events (pipe leaks/valve failures).
   - Sudden consumption spikes.
   - Low tank storage stress scenarios.
