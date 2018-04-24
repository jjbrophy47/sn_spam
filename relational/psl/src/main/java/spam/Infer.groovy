package spam

// Java imports
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
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
// inference
import org.linqs.psl.application.inference.LazyMPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult
// evaluation
import org.linqs.psl.utils.evaluation.statistics.RankingScore
import org.linqs.psl.utils.evaluation.statistics.SimpleRankingComparator

/**
 * Infer relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Infer {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"
    private static final String L_PT = "labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m
    private PrintWriter fw

    /**
     * Constructor.
     *
     * @param data_f folder to store temporary datastore in.
     */
    public Infer(String data_f, status_f, fold) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = data_f + 'db/psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        // this.fw = new PrintWriter(System.out)
    }

    private void out(String message, def newline=1) {
        String msg = newline == 1 ? '\n' + message : message
        // this.fw.print(msg)
        // this.fw.flush()
    }

    private void time(long t1, def suffix='m') {
        long elapsed = System.currentTimeMillis() - t1

        if (suffix == 's') {
            elapsed /= 1000.0
        }
        else if (suffix == 'm') {
            elapsed /= (1000.0 * 60.0)
        }
        else if (suffix == 'h') {
            elapsed /= (1000.0 * 60.0 * 60)
        }

        // out(elapsed.toString() + suffix, 0)
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
        m.add predicate: "inapp", types: [unique_id, unique_id]
        m.add predicate: "inchannel", types: [unique_id, unique_id]
        m.add predicate: "hasip", types: [unique_id, unique_id]
        m.add predicate: "hasos", types: [unique_id, unique_id]
        m.add predicate: "hasdevice", types: [unique_id, unique_id]
        m.add predicate: "hasusrapp", types: [unique_id, unique_id]
        m.add predicate: "inr0", types: [unique_id, unique_id]
        m.add predicate: "inr1", types: [unique_id, unique_id]
        m.add predicate: "inr2", types: [unique_id, unique_id]
        m.add predicate: "inr3", types: [unique_id, unique_id]
        m.add predicate: "inr4", types: [unique_id, unique_id]
        m.add predicate: "inr5", types: [unique_id, unique_id]
        m.add predicate: "inr6", types: [unique_id, unique_id]
        m.add predicate: "inr7", types: [unique_id, unique_id]
        m.add predicate: "insex", types: [unique_id, unique_id]
        m.add predicate: "inage", types: [unique_id, unique_id]
        m.add predicate: "intimepassed", types: [unique_id, unique_id]
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
        m.add predicate: "spammyapp", types: [unique_id]
        m.add predicate: "spammychannel", types: [unique_id]
        m.add predicate: "spammyip", types: [unique_id]
        m.add predicate: "spammyos", types: [unique_id]
        m.add predicate: "spammydevice", types: [unique_id]
        m.add predicate: "spammyusrapp", types: [unique_id]
        m.add predicate: "spammyr0", types: [unique_id]
        m.add predicate: "spammyr1", types: [unique_id]
        m.add predicate: "spammyr2", types: [unique_id]
        m.add predicate: "spammyr3", types: [unique_id]
        m.add predicate: "spammyr4", types: [unique_id]
        m.add predicate: "spammyr5", types: [unique_id]
        m.add predicate: "spammyr6", types: [unique_id]
        m.add predicate: "spammyr7", types: [unique_id]
        m.add predicate: "spammysex", types: [unique_id]
        m.add predicate: "spammyage", types: [unique_id]
        m.add predicate: "spammytimepassed", types: [unique_id]
    }

    /**
     * Load model rules from a text file.
     *
     *@param filename name of the text file with the model rules.
     */
    private void define_rules(String filename) {
        m.addRules(new FileReader(filename))
        // out(m.toString())
    }

    /**
     * Load validation and training predicate data.
     *
     *@param fold experiment identifier.
     *@param data_f folder to load data from.
     */
    private void load_data(int fold, String data_f) {
        long start = System.currentTimeMillis()

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)
        Partition labels_pt = this.ds.getPartition(L_PT)

        // load test set comments to be labeled.
        load_file(data_f + 'test_' + fold, spam, labels_pt)
        load_file(data_f + 'test_no_label_' + fold, spam, write_pt)
        load_file(data_f + 'test_pred_' + fold, indPred, read_pt)

        // load relational data.
        load_file(data_f + 'test_intext_' + fold, intext, read_pt)
        load_file(data_f + 'test_text_' + fold, spammytext, write_pt)

        load_file(data_f + 'test_posts_' + fold, posts, read_pt)
        load_file(data_f + 'test_user_' + fold, spammyuser, write_pt)

        load_file(data_f + 'test_intrack_' + fold, intrack, read_pt)
        load_file(data_f + 'test_track_' + fold, spammytrack, write_pt)

        load_file(data_f + 'test_inhash_' + fold, inhash, read_pt)
        load_file(data_f + 'test_hash_' + fold, spammyhash, write_pt)

        load_file(data_f + 'test_inment_' + fold, inment, read_pt)
        load_file(data_f + 'test_ment_' + fold, spammyment, write_pt)

        load_file(data_f + 'test_invideo_' + fold, invideo, read_pt)
        load_file(data_f + 'test_video_' + fold, spammyvideo, write_pt)

        load_file(data_f + 'test_inhour_' + fold, inhour, read_pt)
        load_file(data_f + 'test_hour_' + fold, spammyhour, write_pt)

        load_file(data_f + 'test_inlink_' + fold, inlink, read_pt)
        load_file(data_f + 'test_link_' + fold, spammylink, write_pt)

        load_file(data_f + 'test_inhotel_' + fold, inhotel, read_pt)
        load_file(data_f + 'test_hotel_' + fold, spammyhotel, write_pt)

        load_file(data_f + 'test_inrest_' + fold, inrest, read_pt)
        load_file(data_f + 'test_rest_' + fold, spammyrest, write_pt)

        load_file(data_f + 'test_inapp_' + fold, inapp, read_pt)
        load_file(data_f + 'test_app_' + fold, spammyapp, write_pt)

        load_file(data_f + 'test_inchannel_' + fold, inchannel, read_pt)
        load_file(data_f + 'test_channel_' + fold, spammychannel, write_pt)

        load_file(data_f + 'test_hasip_' + fold, hasip, read_pt)
        load_file(data_f + 'test_ip_' + fold, spammyip, write_pt)

        load_file(data_f + 'test_hasos_' + fold, hasos, read_pt)
        load_file(data_f + 'test_os_' + fold, spammyos, write_pt)

        load_file(data_f + 'test_hasdevice_' + fold, hasdevice, read_pt)
        load_file(data_f + 'test_device_' + fold, spammydevice, write_pt)

        load_file(data_f + 'test_hasusrapp_' + fold, hasusrapp, read_pt)
        load_file(data_f + 'test_usrapp_' + fold, spammyusrapp, write_pt)

        load_file(data_f + 'test_inr0_' + fold, inr0, read_pt)
        load_file(data_f + 'test_r0_' + fold, spammyr0, write_pt)

        load_file(data_f + 'test_inr1_' + fold, inr1, read_pt)
        load_file(data_f + 'test_r1_' + fold, spammyr1, write_pt)

        load_file(data_f + 'test_inr2_' + fold, inr2, read_pt)
        load_file(data_f + 'test_r2_' + fold, spammyr2, write_pt)

        load_file(data_f + 'test_inr3_' + fold, inr3, read_pt)
        load_file(data_f + 'test_r3_' + fold, spammyr3, write_pt)

        load_file(data_f + 'test_inr4_' + fold, inr4, read_pt)
        load_file(data_f + 'test_r4_' + fold, spammyr4, write_pt)

        load_file(data_f + 'test_inr5_' + fold, inr5, read_pt)
        load_file(data_f + 'test_r5_' + fold, spammyr5, write_pt)

        load_file(data_f + 'test_inr6_' + fold, inr6, read_pt)
        load_file(data_f + 'test_r6_' + fold, spammyr6, write_pt)

        load_file(data_f + 'test_inr7_' + fold, inr7, read_pt)
        load_file(data_f + 'test_r7_' + fold, spammyr7, write_pt)

        load_file(data_f + 'test_insex_' + fold, insex, read_pt)
        load_file(data_f + 'test_sex_' + fold, spammysex, write_pt)

        load_file(data_f + 'test_inage_' + fold, inage, read_pt)
        load_file(data_f + 'test_age_' + fold, spammyage, write_pt)

        load_file(data_f + 'test_intimepassed_' + fold, intimepassed, read_pt)
        load_file(data_f + 'test_timepassed_' + fold, spammytimepassed,
                write_pt)
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
                inhash, inment, invideo, inhour, inlink, inapp, inchannel,
                hasip, hasos, hasdevice]
        return closed
    }

    /**
     * Run inference with the trained model on the test set.
     *
     *@param set of closed predicates.
     *@return a FullInferenceResult object.
     */
    private FullInferenceResult run_inference(Set<Predicate> closed) {
        // out('inference...')
        long start = System.currentTimeMillis()

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        Database inference_db = this.ds.getDatabase(write_pt, closed, read_pt)
        LazyMPEInference mpe = new LazyMPEInference(this.m, inference_db,
                this.cb)
        FullInferenceResult result = mpe.mpeInference()
        mpe.close()
        mpe.finalize()
        inference_db.close()

        time(start)
        return result
    }

    private void evaluate(Set<Predicate> closed) {
        // out('evaluating...')
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

        time(start)

        // out('AUPR: ' + score[0].trunc(4))
        // out(', N-AUPR: ' + score[1].trunc(4), 0)
        // out(', AUROC: ' + score[2].trunc(4), 0)

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
        // out('writing predictions...')
        long start = System.currentTimeMillis()

        Partition temp_pt = this.ds.getPartition('temp_pt')
        Partition write_pt = this.ds.getPartition(W_PT)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        DecimalFormat formatter = new DecimalFormat("#.#####")
        FileWriter fw = new FileWriter(pred_f + 'psl_preds_' + fold + '.csv')

        fw.write('com_id,psl_pred\n')
        for (GroundAtom atom : Queries.getAllAtoms(predictions_db, spam)) {
            double pred = atom.getValue()
            String com_id = atom.getArguments()[0].toString().replace("'", "")
            fw.write(com_id + ',' + formatter.format(pred) + '\n')
        }
        fw.close()
        predictions_db.close()

        time(start)
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param fold experiment identifier.
     *@param iden identifier for subgraph to reason over.
     *@param data_f data folder.
     *@param pred_f predictions folder.
     *@param model_f model folder.
     */
    private void run(int fold, int iden, String data_f, String pred_f,
                     String model_f) {
        String rules_filename = model_f + 'rules_' + fold + '.txt'

        define_predicates()
        define_rules(rules_filename)
        load_data(iden, data_f)
        Set<Predicate> closed = define_closed_predicates()
        FullInferenceResult result = run_inference(closed)
        print_inference_info(result)
        write_predictions(iden, pred_f)

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
        String status_f = '../output/' + domain + '/status/'
        new File(pred_f).mkdirs()
        new File(model_f).mkdirs()
        return new Tuple(data_f, pred_f, model_f, status_f)
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return a tuple containing the experiment id and social network.
     */
    public static Tuple check_commandline_args(String[] args) {
        if (args.length < 3) {
            print('Missing args, example: [fold] [domain] [relations (opt)]')
            System.exit(0)
        }
        int fold = args[0].toInteger()
        int iden = args[1].toInteger()
        String domain = args[2].toString()
        return new Tuple(fold, iden, domain)
    }

    /**
     * Main method that creates and runs the Infer object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (fold, iden, domain) = check_commandline_args(args)
        def (data_f, pred_f, model_f, status_f) = define_file_folders(domain)
        Infer b = new Infer(data_f, status_f, fold)
        b.run(fold, iden, data_f, pred_f, model_f)
    }
}
