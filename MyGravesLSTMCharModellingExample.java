package org.deeplearning4j.examples.recurrent.character;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.ui.api.UIServer;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.optimize.api.IterationListener;
import java.util.ArrayList;
/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	from Project Gutenberg. Training on other text sources should be relatively easy to implement.

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class MyGravesLSTMCharModellingExample {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 20;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 15;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 100;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 80;				//Length of each sample to generate
		String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);
        String[] validCharacters;
        //Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
        File inputFile = new ClassPathResource("hurriyet_training2.txt").getFile();

        validCharacters = MorphIterator.getMinimalCharacterSet(inputFile.getAbsolutePath());
        MorphIterator iter = getIterator(miniBatchSize,exampleLength,  validCharacters);
		int nOut = iter.totalOutcomes();
        File testInputFile = new ClassPathResource("hurtest.txt").getFile();
        MorphIterator testIter=new  MorphIterator(testInputFile.getAbsolutePath(), Charset.forName("UTF-8"),
            miniBatchSize, exampleLength, validCharacters, new Random(12345));
       // MorphIterator testIter= new MorphIterator(miniBatchSize, exampleLength, validCharacters, new Random(12345));
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.01)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()+14).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(2, new RnnOutputLayer.Builder().activation("sigmoid")        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        UIServer uiServer = UIServer.getInstance();

       /* StatsStorage statsStorage = new InMemoryStatsStorage();
        ArrayList<IterationListener> listenerList = new ArrayList<IterationListener>();
        listenerList.add(new StatsListener(statsStorage));
        listenerList.add(new ScoreIterationListener(1));
        net.setListeners(listenerList);

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);*/
		net.setListeners(new ScoreIterationListener(1));

		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		//Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);

            }
            iter.reset();	//Reset iterator for another epoch
            System.out.println("Epoch " + i + " complete. Starting evaluation:");
            Evaluation evaluation = new Evaluation();
            while(testIter.hasNext()) {
                DataSet testds = testIter.next();
                INDArray predicted = net.output(testds.getFeatureMatrix(),false);
                evaluation.evalTimeSeries(testds.getLabels(),predicted);
            }
            //String[] samples = sampleCharactersFromNetwork(generationInitialization,net,testIter,rng,nCharactersToSample,nSamplesToGenerate);

            testIter.reset();
            System.out.println(evaluation.stats());

		}

		System.out.println("\n\nExample complete");
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param sequenceLength Number of characters in each text segment.
	 */
	public static MorphIterator getIterator(int miniBatchSize, int sequenceLength,String[] validCharacters) throws Exception{

        File inputFile = new ClassPathResource("hurriyet_training_ver2.txt").getFile();

        // creating SentenceIterator wrapping our training corpus
		//String fileLocation = "resources/training_sentences.txt";	//Storage location from downloaded file

		File f = new File(inputFile.getAbsolutePath());

		//Which characters are allowed? Others will be removed
		return new MorphIterator(inputFile.getAbsolutePath(), Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random(12345));
	}

	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        MorphIterator iter, Random rng, int charactersToSample, int numSamples )throws Exception{
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = "";//String.valueOf(iter.getRandomCharacter());
		}

        File inputFile = new ClassPathResource("test_sentences2.txt").getFile();
        List<String> lines = Files.readAllLines(inputFile.toPath(),Charset.forName("UTF-8"));
        String st=lines.get(rng.nextInt(lines.size()));
        System.out.println("Test Sentence; ");
        System.out.println(st);
        String delims = "[::]";
        String[] words = st.split(delims);
        String[] myWord = words[0].split("\\s+");
        INDArray initializationInput =Nd4j.zeros(numSamples, iter.inputColumns()+14, words.length);//ilk kelime+ boşluk+yeni kelimenin ilk..


                if(!words[0].isEmpty()) {
                    String delim2 = "[ ]";
                    String[] morphemes = words[0].split(delim2);
                    for( int i=0; i<morphemes.length; i++ ){//for (String morpheme : morphemes) {
                        int idx = iter.convertCharacterToIndex(morphemes[i]);
                        for( int j=0; j<numSamples; j++ ){
                            initializationInput.putScalar(new int[]{j,idx,0}, 1.0);
                        }
                        initialization=initialization.concat(morphemes[i]);
                    }
                }


                initialization=initialization.concat(Character.toString(' '));
            if(!words[1].isEmpty()) {
                String delim2 = "[ ]";
                String[] morphemes = words[1].split(delim2);

                   int idx = iter.convertCharacterToIndex(morphemes[0]);
                    for( int j=0; j<numSamples; j++ ){
                        initializationInput.putScalar(new int[]{j,idx+iter.inputColumns(),0}, 1.0);
                    }
                   initialization=initialization.concat(morphemes[0]);

            }


		//Create input for initialization
		//INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
	/*	char[] init = initialization.toCharArray();
		for( int i=0; i<init.length; i++ ){
			int idx = iter.convertCharacterToIndex(init[i]);
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
			}
		}*/

		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}

			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
