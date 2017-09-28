YouTube Dataset
===

Place `comments.csv` file here.

### Attributes (6): ###

* *com_id*: unique int.
* *timestamp*: time the comment was posted.
* *vid_id*: video the comment was posted on.
* *user_id*: user who posted the comment.
* *text*: content of the comment.
* *label*: 0 - not spam, 1 - spam.

---

### Basic Statistics ###

* *6,431,471* total comments; *481,334* spam comments (7.5%).
* *2,860,264* users; *177,542* spammers (6.2%).
* *6,407* videos; *6,340* spam videos (98.9%).

---

### Running Times ###

These running times are to give you a sense of how long each operation could take. Individual running times may vary. Models are run on a single linux RedHat--Santiago 6.9--machine at 2.67gHz with 12 cores and 72gb RAM.

#### Independent Model ####

- training (85%): **4m**, testing (15%): **0.4m**.
	* feature construction: **25m**.

#### Relational Model ####

Relations used: *posts*, *text*.

##### Training (validation set size): #####
- 321,574 comments (5%) -- 846,778 nodes: **8m**
- 643,148 comments (10%) -- 1,745,171 nodes: **19m**
- 964,721 comments (15%) -- 2,653,013 nodes: **27m**

##### Inference (testing set size): ######
- 964,721 comments (15%) -- 2,635,303 nodes: **8m**