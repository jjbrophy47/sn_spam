# Social Network Spam #

Applying collective classification to identify spam.

## Wiki ##

Check out the [project wiki!](https://bitbucket.org/jbrophy/spam/wiki/Home).

## Data ##

The data is not included in this repository. Once the data is downloaded, do the following:

1. Place the `comments.csv` file in the appropriate data folder: `independent/data/<domain>/`.
2. If there is a user graph associated with this data, place the `network.tsv` file in the data folder as well.
	* If you already possess the `graph_features.csv` file, you can place that in the features folder: `independent/output/<domain>/features/`.

There are readme files in each of these folders to indicate where to put these data files.

## Reasoning Engines ##

* *PSL*: There is a readme file in the `relational/psl/` directory with instructions on how to install the necessary components of PSL to run this application.

* *Tuffy*: There is a readme file in the `relational/tuffy/` directory with instructions on how to install the necessary components of Tuffy to run this application.

### Contact Info ###

Jonathan Brophy {jbrophy@cs.uoregon.edu}