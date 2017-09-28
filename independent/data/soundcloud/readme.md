SoundCloud Dataset
===

Place `comments.csv` file here.

### Attributes (6): ###

* *com_id*: unique int.
* *user_id*: user who posted the comment.
* *track_id*: track where the comment was posted.
* *timestamp*: time the comment was posted.
* *text*: content of the comment.
* *label*: 0 - not spam, 1 - spam.

---

### Basic Statistics ###

* *42,783,305* total comments; *684,338* spam comments (1.6%).
* *5,505,634* users; *128,016* spammers (2.3%).
* *7,794,029* tracks; *190,545* spam tracks (2.4%).

---

### Running Times ###

These running times are to give you a sense of how long each operation could take. Individual running times may vary. Models are run on a single linux RedHat--Santiago 6.9--machine at 2.67gHz with 12 cores and 72gb RAM.

#### Independent Model ####

- training (85%): **25m**, testing (15%): **1.3m**.
	* feature construction: **151m**.

#### Relational Model ####

Relations used: *posts*, *text*, *tracks*.

##### Training (validation set size): #####
- 427,833 comments (1%) -- 1,715,354 nodes: **19m**
- 600k (1.4%): **25m**
- 1.2m (2.8%): **59m**
- 2.4m (5.6%): **MEMORY OVERLOAD**

##### Inference (testing set size): ######
- 300k (0.7%): **4m**
- 600k (1.4%): **8m**
- 1.2m (2.4%): **17m**
- 2.4m (4.8%): **49m**
- 4.8m (9.6%): **422m**
- 6.6m (15%): **>24h**