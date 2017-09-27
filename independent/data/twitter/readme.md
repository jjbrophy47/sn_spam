Twitter Dataset
===

Place `comments.csv` file here.

### Attributes (6): ###

* *com_id*: unique int.
* *user_id*: user who posted the comment.
* *text*: content of the comment.
* *label*: 0 - not spam, 1 - spam.

---

### Basic Statistics ###

* *8,845,979* total comments; *1,722,144* spam comments (19.5%).
* *4,831,679* users; *843,002* spammers (17.5%).

---

### Running Times ###

These running times are to give you a sense of how long each operation could take. Individual running times may vary. Models are run on a single linux RedHat--Santiago 6.9--machine at 2.67gHz with 12 cores and 72gb RAM.

#### Independent Model ####

- training (85%): **7m**, testing (15%): **0.7m**.
	* feature construction: **45m**.

#### Relational Model ####

Relations used: *posts*, *text*, *hashtags*, *mentions*.

##### Training (validation set size): #####
- 300k (3.4%): **11m**
- 600k (6.8%): **28m**
- 1.2m (13.6%): **73m**
- 1,326,897 (15%): **94m**

##### Inference (testing set size): ######
- 1,326,897 (15%): ~??