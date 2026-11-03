# Bank Fraud Detection

A mini project that compares fraud-detection performance for 3 feature groups:

- **Transaction counts per category:** transactions are counted during the previous 1, 2, 5, and 10 calendar months (1 month back from 10th of April will take transaction back to 10th of March). The counts are stored by reference date and joined to the training examples.
- **Phone and account country match:** compares the country where the phone number is registered with the country prefix of the customer’s IBAN. (`1` for a match, `0` for a mismatch, and `NaN` when one or both contries are unknown)

THe experiments compare precision, recall, and F1 of:
- baseline
- baseline with each new feature
- all features together -> improved F1 by 4% compared to baseline

Tests with 100% coverage run automatically through GitHub Actions