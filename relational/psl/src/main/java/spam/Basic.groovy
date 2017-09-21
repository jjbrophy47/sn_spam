package spam

// Java imports
import java.io.File
import java.io.FileWriter
import java.text.DecimalFormat
// PSL imports
import org.linqs.psl.config.ConfigBundle
import org.linqs.psl.config.ConfigManager
// database
import org.linqs.psl.database.Partition
import org.linqs.psl.database.DataStore
import org.linqs.psl.database.Database
import org.linqs.psl.database.Queries
import org.linqs.psl.database.loading.Inserter
import org.linqs.psl.database.rdbms.RDBMSDataStore
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type
// data loading
import org.linqs.psl.utils.dataloading.InserterUtils
// model
import org.linqs.psl.groovy.PSLModel
import org.linqs.psl.model.rule.Rule
import org.linqs.psl.model.atom.GroundAtom
import org.linqs.psl.model.term.ConstantType
import org.linqs.psl.model.predicate.Predicate
// weight learning
import org.linqs.psl.application.learning.weight.em.HardEM
// inference
import org.linqs.psl.application.inference.MPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult
// evaluation
import org.linqs.psl.utils.evaluation.statistics.RankingScore
import org.linqs.psl.utils.evaluation.statistics.SimpleRankingComparator

/**
 * Basic relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Basic {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"
    private static final String L_PT = "labels_pt"
    private static final String WL_W_PT = "wl_write_pt"
    private static final String WL_R_PT = "wl_read_pt"
    private static final String WL_L_PT = "wl_labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m

    /**
     * Constructor.
     *
     * @param data_f folder to store temporary datastore in.
     */
    public Basic(String data_f) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = data_f + 'db/psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        print('data store setup at: ' + db_path)
    }

    /**
     * Specify and add predicate definitions to the model.
     */
    private void define_predicates() {
        ConstantType unique_id = ConstantType.UniqueID
        m.add predicate: "spam", types: [unique_id]
        m.add predicate: "indpred", types: [unique_id]
        m.add predicate: "intext", types: [unique_id, unique_id]
        m.add predicate: "posts", types: [unique_id, unique_id]
        m.add predicate: "intrack", types: [unique_id, unique_id]
        m.add predicate: "inhash", types: [unique_id, unique_id]
        m.add predicate: "inment", types: [unique_id, unique_id]
        m.add predicate: "invideo", types: [unique_id, unique_id]
        m.add predicate: "inhour", types: [unique_id, unique_id]
        m.add predicate: "inlink", types: [unique_id, unique_id]
        m.add predicate: "inhotel", types: [unique_id, unique_id]
        m.add predicate: "inrest", types: [unique_id, unique_id]
        m.add predicate: "spammytext", types: [unique_id]
        m.add predicate: "spammyuser", types: [unique_id]
        m.add predicate: "spammytrack", types: [unique_id]
        m.add predicate: "spammyhash", types: [unique_id]
        m.add predicate: "spammyment", types: [unique_id]
        m.add predicate: "spammyvideo", types: [unique_id]
        m.add predicate: "spammyhour", types: [unique_id]
        m.add predicate: "spammylink", types: [unique_id]
        m.add predicate: "spammyhotel", types: [unique_id]
        m.add predicate: "spammyrest", types: [unique_id]
    }

    /**
     * Load model rules from a text file.
     *
     *@param filename name of the text file with the model rules.
     */
    private void define_rules(String filename) {
        print('\nloading model...')
        long start = System.currentTimeMillis()
        m.addRules(new FileReader(filename))
        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')
    }

    /**
     * Load validation and training predicate data.
     *
     *@param fold experiment identifier.
     *@param data_f folder to load data from.
     */
    private void load_data(int fold, String data_f) {
        print('\nloading data...')
        long start = System.currentTimeMillis()

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)
        Partition wl_write_pt = this.ds.getPartition(WL_W_PT)
        Partition wl_read_pt = this.ds.getPartition(WL_R_PT)
        Partition labels_pt = this.ds.getPartition(L_PT)
        Partition wl_labels_pt = this.ds.getPartition(WL_L_PT)

        // load test set comments to be labeled.
        load_file(data_f + 'test_' + fold, spam, labels_pt)
        load_file(data_f + 'test_no_label_' + fold, spam, write_pt)
        load_file(data_f + 'test_pred_' + fold, indPred, read_pt)
        load_file(data_f + 'val_' + fold, spam, wl_labels_pt)
        load_file(data_f + 'val_no_label_' + fold, spam, wl_write_pt)
        load_file(data_f + 'val_pred_' + fold, indPred, wl_read_pt)

        // load relational data.
        load_file(data_f + 'test_intext_' + fold, intext, read_pt)
        load_file(data_f + 'test_text_' + fold, spammytext, write_pt)
        load_file(data_f + 'val_intext_' + fold, intext, wl_read_pt)
        load_file(data_f + 'val_text_' + fold, spammytext, wl_write_pt)

        load_file(data_f + 'test_posts_' + fold, posts, read_pt)
        load_file(data_f + 'test_user_' + fold, spammyuser, write_pt)
        load_file(data_f + 'val_posts_' + fold, posts, wl_read_pt)
        load_file(data_f + 'val_user_' + fold, spammyuser, wl_write_pt)

        load_file(data_f + 'test_intrack_' + fold, intrack, read_pt)
        load_file(data_f + 'test_track_' + fold, spammytrack, write_pt)
        load_file(data_f + 'val_intrack_' + fold, intrack, wl_read_pt)
        load_file(data_f + 'val_track_' + fold, spammytrack, wl_write_pt)

        load_file(data_f + 'test_inhash_' + fold, inhash, read_pt)
        load_file(data_f + 'test_hash_' + fold, spammyhash, write_pt)
        load_file(data_f + 'val_inhash_' + fold, inhash, wl_read_pt)
        load_file(data_f + 'val_hash_' + fold, spammyhash, wl_write_pt)

        load_file(data_f + 'test_inment_' + fold, inment, read_pt)
        load_file(data_f + 'test_ment_' + fold, spammyment, write_pt)
        load_file(data_f + 'val_inment_' + fold, inment, wl_read_pt)
        load_file(data_f + 'val_ment_' + fold, spammyment, wl_write_pt)

        load_file(data_f + 'test_invideo_' + fold, invideo, read_pt)
        load_file(data_f + 'test_video_' + fold, spammyvideo, write_pt)
        load_file(data_f + 'val_invideo_' + fold, invideo, wl_read_pt)
        load_file(data_f + 'val_video_' + fold, spammyvideo, wl_write_pt)

        load_file(data_f + 'test_inhour_' + fold, inhour, read_pt)
        load_file(data_f + 'test_hour_' + fold, spammyhour, write_pt)
        load_file(data_f + 'val_inhour_' + fold, inhour, wl_read_pt)
        load_file(data_f + 'val_hour_' + fold, spammyhour, wl_write_pt)

        load_file(data_f + 'test_inlink_' + fold, inlink, read_pt)
        load_file(data_f + 'test_link_' + fold, spammylink, write_pt)
        load_file(data_f + 'val_inlink_' + fold, inlink, wl_read_pt)
        load_file(data_f + 'val_link_' + fold, spammylink, wl_write_pt)

        load_file(data_f + 'test_inhotel_' + fold, inhotel, read_pt)
        load_file(data_f + 'test_hotel_' + fold, spammyhotel, write_pt)
        load_file(data_f + 'val_inhotel_' + fold, inhotel, wl_read_pt)
        load_file(data_f + 'val_hotel_' + fold, spammyhotel, wl_write_pt)

        load_file(data_f + 'test_inrest_' + fold, inrest, read_pt)
        load_file(data_f + 'test_rest_' + fold, spammyrest, write_pt)
        load_file(data_f + 'val_inrest_' + fold, inrest, wl_read_pt)
        load_file(data_f + 'val_rest_' + fold, spammyrest, wl_write_pt)

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')
    }

    /**
     * Loads a tab separated predicate data file. Automatically handles
     * truth and non truth files.
     *
     *@param filename name of the file to load.
     *@param predicate name of the predicate to load data for.
     *@param partition parition to load the file into.
     */
    private void load_file(filename, predicate, partition) {
        String file = filename + '.tsv'
        if (new File(file).exists()) {
            Inserter inserter = this.ds.getInserter(predicate, partition)
            InserterUtils.loadDelimitedDataAutomatic(predicate, inserter, file)
        }
    }

    /**
     * Specifies which predicates are closed (i.e. observations that cannot
     * be changed).
     *
     *@return a set of closed predicates.
     */
    private Set<Predicate> define_closed_predicates() {
        Set<Predicate> closed = [indpred, intext, posts, intrack,
                inhash, inment, invideo, inhour, inlink]
        return closed
    }

    /**
     * Learn weights for model rules using vlidation data.
     *
     *@param closed set of closed predicates.
     */
    private void learn_weights(Set<Predicate> closed) {
        Set<Predicate> closed_labels = [spam]

        Partition wl_wr_pt = ds.getPartition(WL_W_PT)
        Partition wl_r_pt = ds.getPartition(WL_R_PT)
        Partition wl_l_pt = ds.getPartition(WL_L_PT)

        print('\nlearning weights...')
        long start = System.currentTimeMillis()

        Database wl_tr_db = this.ds.getDatabase(wl_wr_pt, closed, wl_r_pt)
        Database wl_l_db = ds.getDatabase(wl_l_pt, closed_labels)

        HardEM w_learn = new HardEM(this.m, wl_tr_db, wl_l_db, this.cb)
        w_learn.learn()
        wl_tr_db.close()
        wl_l_db.close()

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')
    }

    /**
     * Write the model with learned weights to a text file.
     *
     *@param fold experiment identifier.
     *@param model_f folder to save model to.
     */
    private void write_model(int fold, String model_f) {
        FileWriter mw = new FileWriter(model_f + 'rules_' + fold + '.txt')
        for (Rule rule : this.m.getRules()) {
            String rule_str = rule.toString().replace('~( ', '~')
            String rule_filtered = rule_str.replace('( ', '').replace(' )', '')
            print('\n\t' + rule_str)
            mw.write(rule_filtered + '\n')
        }
        mw.close()
    }

    /**
     * Run inference with the trained model on the test set.
     *
     *@param set of closed predicates.
     *@return a FullInferenceResult object.
     */
    private FullInferenceResult run_inference(Set<Predicate> closed) {
        print('\nrunning inference...')
        long start = System.currentTimeMillis()

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        Database inference_db = this.ds.getDatabase(write_pt, closed, read_pt)
        MPEInference mpe = new MPEInference(this.m, inference_db, this.cb)
        FullInferenceResult result = mpe.mpeInference()
        mpe.close()
        mpe.finalize()
        inference_db.close()

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')

        return result
    }

    private void evaluate(Set<Predicate> closed) {
        print('\nevaluating...')
        long start = System.currentTimeMillis()

        Partition labels_pt = this.ds.getPartition(L_PT)
        Partition write_pt = this.ds.getPartition(W_PT)
        Partition temp_pt = this.ds.getPartition('evaluation_pt')

        Database labels_db = this.ds.getDatabase(labels_pt, closed)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        def comparator = new SimpleRankingComparator(predictions_db)
        comparator.setBaseline(labels_db)

        def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,
                RankingScore.AreaROC]
        double[] score = new double[metrics.size()]

        for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(spam)
        }

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')

        print('\n\tAUPR: ' + score[0].trunc(4))
        print(', N-AUPR: ' + score[1].trunc(4))
        print(', AUROC: ' + score[2].trunc(4))

        labels_db.close()
        predictions_db.close()
    }

    /**
     * Print inference result information.
     *
     *@param r object resulting from inference.
     */
    private void print_inference_info(FullInferenceResult r) {
        float incomp = r.getTotalWeightedIncompatibility().trunc(2)
        int grnd_atoms = r.getNumGroundAtoms()
        int grnd_evd = r.getNumGroundEvidence()
        def s = 'incompatibility: ' + incomp.toString()
        s += ', ground atoms: ' + grnd_atoms.toString()
        s += ', ground evidence: ' + grnd_evd.toString()
    }

    /**
     * Write the relational model predictions for each comment in the test set.
     *
     *@param fold experiment identifier.
     *@param pred_f folder to save predictions to.
     */
    private void write_predictions(int fold, String pred_f) {
        print('\nwriting predictions...')
        long start = System.currentTimeMillis()

        Partition temp_pt = this.ds.getPartition('temp_pt')
        Partition write_pt = this.ds.getPartition(W_PT)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        DecimalFormat formatter = new DecimalFormat("#.#####")
        FileWriter fw = new FileWriter(pred_f + 'predictions_' + fold + '.csv')

        fw.write('com_id,rel_pred\n')
        for (GroundAtom atom : Queries.getAllAtoms(predictions_db, spam)) {
            double pred = atom.getValue()
            String com_id = atom.getArguments()[0].toString().replace("'", "")
            fw.write(com_id + ',' + formatter.format(pred) + '\n')
        }
        fw.close()
        predictions_db.close()

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's\n')
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param fold experiment identifier.
     *@param data_f data folder.
     *@param pred_f predictions folder.
     *@param model_f model folder.
     */
    private void run(int fold, String data_f, String pred_f, String model_f) {
        String rules_filename = data_f + 'rules_' + fold + '.txt'

        define_predicates()
        define_rules(rules_filename)
        load_data(fold, data_f)
        Set<Predicate> closed = define_closed_predicates()
        learn_weights(closed)
        write_model(fold, model_f)
        FullInferenceResult result = run_inference(closed)
        print_inference_info(result)
        evaluate(closed)
        write_predictions(fold, pred_f)

        this.ds.close()
    }

    /**
     * Specifies relative paths to 'psl' directory.
     *
     *@param domain social network (e.g. soundcloud, youtube, twitter, etc.).
     *@return a tuple with various data folder paths.
     */
    public static Tuple define_file_folders(String domain) {
        String data_f = './data/' + domain + '/'
        String pred_f = '../output/' + domain + '/predictions/'
        String model_f = '../output/' + domain + '/models/'
        new File(pred_f).mkdirs()
        new File(model_f).mkdirs()
        return new Tuple(data_f, pred_f, model_f)
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return a tuple containing the experiment id and social network.
     */
    public static Tuple check_commandline_args(String[] args) {
        if (args.length < 2) {
            print('Missing args, example: [fold] [domain] [relations (opt)]')
            System.exit(0)
        }
        int fold = args[0].toInteger()
        String domain = args[1].toString()
        return new Tuple(fold, domain)
    }

    /**
     * Main method that creates and runs the Basic object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (fold, domain) = check_commandline_args(args)
        def (data_f, pred_f, model_f) = define_file_folders(domain)
        Basic b = new Basic(data_f)
        b.run(fold, data_f, pred_f, model_f)
    }
}
