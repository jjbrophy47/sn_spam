IfWe Dataset
===

Place `comments.csv` file here.

### Attributes (5): ###

* *com_id*: unique identifier per user.
* *sex*: 1 - Male, 2 - Female.
* *time_passed_validation*: Normalized ([0, 1]) amount of time for user to be validated.
* *age_group*: 1 (10-20), 2 (20-30), 3 (30-40), etc.
* *label*: 0 - not spam, 1 - spam.

---

### Basic Statistics ###

* *5,607,447* total users; *336,953* spammers (7.5%).
* *3,788,948* males; *156,642* male spammers (4.1%).
* *1,818,499* females; *180,311* female spammers (9.9%).

---

### Running Times ###

These running times are to give you a sense of how long each operation could take. Individual running times may vary. Models are run on a single linux RedHat--Santiago 6.9--machine at 2.67gHz with 12 cores and 72gb RAM.

#### Independent Model ####

- training (85%): **22m**, testing (15%): **0.2m**.
	* feature construction: **14m**.

#### Relational Model ####

Relations used: *posts*, *text*, *hashtags*, *mentions*.

##### Training (validation set size): #####
- 300k (3.4%): **~??**
- 600k (6.8%): **~??**
- 1.2m (13.6%): **~??**
- (15%): **~??**

##### Inference (testing set size): ######
- (15%): **~??**