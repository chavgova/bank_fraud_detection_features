# uva-mlprod-code
## Week 1 - Data Driven Roles, MLOps Lifecycle, Clean Coding, Unit Testing


### Highlights:

#### Clean Coding
- Lock down dependency versions
- Normally the data is not in git.
- I use config instead of hardcoding
  - I make sure that configs are not mutable so that runtime cannot change it.
  - I break down the configs into multiple groups so easier to manage
  - We may need different configs for dev, staging, prod
- Don't forget: `export PYTHONPATH=~/uva-mlprod-code`
- Don't forget: `. .venv/bin/activate`
- There is an error in z_score because it was difficult to test


#### Unit Testing

- Edge cases matter
- Multiple and complex tests are required. z_score 1st test pass but 2nd fail.
- ML models are stochastic and may be chaotic. It does not make much sense to test training or testing logic.
- We can have multiple cases in a test. Upside: easier to setup tests. Downside: difficult to understand which case failed.


### Clean Code Pointers

- Readable, Testability, Maintainable

- C2: Obsolete Comment
- C5: Commented-out code
- F1: Too-many arguments
- F3: Flag Arguments
- G5: Duplication
- G19: Use explanatory variables
- F20: Function names should say what they do
- G21: Understand the algorithm (*)
- G25: Replace magic numbers with named constants
- G30: Functions should do one thing
- N1: Choose descriptive names
