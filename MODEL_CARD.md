# Model Card: Hotel Cancellation Classifier

## Model Details
- **Model Architecture**: XGBoost Classifier wrapped in `CalibratedClassifierCV(method='sigmoid')`
- **Model Type**: Binary Classification
- **Version**: 2.0.0
- **Training Algorithm**: Gradient Boosting over temporal walk-forward cross-validation (`TimeSeriesSplit`).

## Intended Use
- **Primary Use Case**: Predicting the likelihood of a hotel booking cancellation at the time of reservation to inform dynamic pricing, overbooking limits, and revenue management.
- **Out of Scope**: Not intended to automatically cancel user reservations or penalize users without human oversight.

## Training Data
- **Dataset**: Antonio, Almeida, and Nunes (2019) Hotel Booking Demand Dataset.
- **Data Remediation**: The original dataset contains target leakage. Columns such as `reservation_status`, `booking_changes`, and `days_in_waiting_list` deterministically leak the cancellation outcome. These columns were **explicitly excluded** from the training features to ensure realistic production bounds.
- **Data Splitting**: Ordered temporally by `arrival_date` to prevent data leakage from the future into the past.

## Evaluation Metrics
Evaluated on a temporal holdout set:
- **AUC (ROC)**: 0.814 (Calibrated: 0.809)
- **Brier Score**: 0.163
- **Calibration**: The model probabilities are tightly calibrated, ensuring that a 0.70 predicted probability corresponds to a true 70% real-world cancellation rate.

## Subgroup Fairness & Analysis
Subgroup metrics are tracked to ensure fairness and consistent performance across operational bounds:
- **By Hotel Type**: Resort Hotel (AUC=0.799), City Hotel (AUC=0.814)
- **By Top 5 Countries**: PRT (0.908), GBR (0.817), FRA (0.743), DEU (0.798), ESP (0.721)

*Note: The model performs exceptionally well on domestic (PRT) bookings, but sees degraded performance for certain international segments like ESP (Spain). Further feature engineering may be needed for specific geos.*

## Known Failure Modes
- Exogenously driven mass cancellations (e.g., pandemics, global travel restrictions) will invalidate the temporal priors.
- Extreme outlier lead times (>1 year) have higher variance in probability calibration.
