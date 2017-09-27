IfWe Dataset
===

Place `comments.csv` file here.

### Attributes (6): ###

* *com_id*: unique identifier per user.
* *label*: 0 - not spam, 1 - spam.
* **TODO**

---

### Basic Statistics ###

* *5,607,447* total users; *336,953* spammers (7.5%).
* *3,788,948* males; *156,642* male spammers (4.1%).
* *1,818,499* females; *180,311* female spammers (9.9%).

---

### Running Times ###

These running times are to give you a sense of how long each operation could take. Individual running times may vary. Models are run on a single linux RedHat--Santiago 6.9--machine at 2.67gHz with 12 cores and 72gb RAM.

#### Independent Model ####

- training (85%): **??**, testing (15%): **??**.
	* feature construction: **??**.

#### Relational Model ####

Relations used: *posts*, *text*, *hashtags*, *mentions*.

##### Training (validation set size): #####
- 300k (3.4%): **~??**
- 600k (6.8%): **~??**
- 1.2m (13.6%): **~??**
- (15%): **~??**

##### Inference (testing set size): ######
- (15%): **~??**