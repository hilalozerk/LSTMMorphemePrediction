package org.deeplearning4j.examples.recurrent.character;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class MorphIterator implements DataSetIterator {
    //Valid characters
	private char[] validCharacters;
    private String[] validMorphemes;
    //Maps each character to an index ind the input/output
	private Map<Character,Integer> charToIdxMap;
    private Map<String,Integer> strToIdxMap;
    //All characters of the input file (after filtering to only those that are valid
	private String[] fileCharacters;
    private INDArray fileWordsArray;
    private INDArray fileTestArray;

    //Length of each example/minibatch (number of characters)
	private int exampleLength;
    //Size of each minibatch (number of examples)
	private int miniBatchSize;
	private Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();
    private LinkedList<Integer> exampleTestStartOffsets = new LinkedList<>();

	/**
	 * @param textFilePath Path to text file to use for generating samples
	 * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
	 * @param miniBatchSize Number of examples per mini-batch
	 * @param exampleLength Number of characters in each input/output vector
	 * @param validMorphemes Character array of valid characters. Characters not present in this array will be removed
	 * @param rng Random number generator, for repeatability if required
	 * @throws IOException If text file cannot  be loaded
	 */
	public MorphIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                         String[] validMorphemes, Random rng) throws IOException {
		if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		this.validCharacters = validCharacters;
        this.validMorphemes=validMorphemes;
		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
		this.rng = rng;

		//Store valid characters is a map for later use in vectorization
		strToIdxMap = new HashMap<>();
		for( int i=0; i<validMorphemes.length; i++ ) strToIdxMap.put(validMorphemes[i], i);

		//Load file and convert contents to a char[]
	/*	List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
		int maxSize = lines.size()+1;	//add lines.size() to account for newline characters at end of each line
		for( String s : lines ){
           // StringTokenizer st = new StringTokenizer(s);
           // maxSize += st.countTokens();
            String delimSentence = "[||]";
            String[] sentences = s.split(delimSentence);
            String delims = "[::]";
            if(sentences.length>4) {
                String[] words = sentences[4].split(delims);
                //maxSize+=(words.length/2)+1;
               // maxSize+=words.length;
                for (String word : words) {
                    if(!word.isEmpty()) {
                       // String delim2 = "[ ]";
                       // String[] morphemes = word.split(delim2);
                       // maxSize+=morphemes.length;
                        maxSize+=1;
                    }

                    //characters[currIdx++] = Character.toString(' ');
                }
                //maxSize+=1;
            }
        }
*/
        ///
     //  File inputFile = new ClassPathResource("hurriyet_training.txt").getFile();
        int maxSize = 0;
        List<String> linesSecondFile = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
        //maxSize+=linesSecondFile.size();
        for( String s : linesSecondFile ){
            String delims= "[::]";
            String[] words = s.split(delims);
            for (String word : words) {
                if(!word.isEmpty()) {

                    maxSize+=1;
                }

            }
           // maxSize+=1;
        }

        ///
	//	String[] characters = new String[maxSize];
        INDArray wordsArray =Nd4j.zeros(maxSize,validMorphemes.length);

		int currIdx = 0;
		/*for( String s : lines ){
            String delimSentence = "[||]";
            String[] sentences = s.split(delimSentence);
                    String delims = "[::]";
            if(sentences.length>4) {
                String[] words = sentences[4].split(delims);
                INDArray wordMorphemes =Nd4j.zeros(1,validMorphemes.length);
                for (String word : words) {
                    if(!word.isEmpty()) {
                        //int[] wordMorphemes=new int[validMorphemes.length];
                        String delim2 = "[ ]";
                        String[] morphemes = word.split(delim2);
                        //String processedWord="";
                        for (String morpheme : morphemes) {
                            if (!strToIdxMap.containsKey(morpheme)) continue;
                            wordMorphemes.putScalar(new int[]{strToIdxMap.get(morpheme)},1);
                            //characters[currIdx++] = morpheme;// if (currIdx < characters.length)
                            //processedWord.concat(morpheme+" ");
                        }
                        wordsArray.putRow(currIdx++,wordMorphemes);
                    //    characters[currIdx++] = processedWord;
                    }
                   // characters[currIdx++] = Character.toString(' ');
                }

            }
           // if (currIdx < characters.length)
            // INDArray nlMorph =Nd4j.zeros(1,validMorphemes.length);
            // nlMorph.putScalar(new int[]{strToIdxMap.get("\n")},1);
            // wordsArray.putRow(currIdx++,nlMorph);
			// characters[currIdx++] = Character.toString('\n');
		}*/
//Second File
        for( String s : linesSecondFile ){
            String delims = "[::]";
            String[] words = s.split(delims);
            INDArray wordMorphemes =Nd4j.zeros(1,validMorphemes.length);
                for (String word : words) {
                    if(!word.isEmpty()) {
                        String delim2 = "[ ]";
                        String[] morphemes = word.split(delim2);
                        //String processedWord="";
                        for (String morpheme : morphemes) {
                            if (!strToIdxMap.containsKey(morpheme)) continue;
                           // characters[currIdx++] = morpheme;// if (currIdx < characters.length)
                            wordMorphemes.putScalar(new int[]{strToIdxMap.get(morpheme)},1);
                            //processedWord.concat(morpheme+" ");
                        }
                        wordsArray.putRow(currIdx++,wordMorphemes);
                        //characters[currIdx++] = processedWord;
                    }
                    //characters[currIdx++] = Character.toString(' ');
                }

            // if (currIdx < characters.length)
            /*INDArray nlMorph =Nd4j.zeros(1,validMorphemes.length);
            nlMorph.putScalar(new int[]{strToIdxMap.get("\n")},1);
            wordsArray.putRow(currIdx++,nlMorph);*/
            //characters[currIdx++] = Character.toString('\n');
        }


        ///
		if( currIdx == wordsArray.size(0) ){
			//fileCharacters = characters;
            fileWordsArray=wordsArray;
		} else {

            int[] karray= new int[currIdx];
            for(int k=0;k< currIdx ;k++){
                karray[k]=k;
            }
            fileWordsArray=wordsArray.getRows(karray);
		}
		fileWordsArray.size(0) ;
		if( exampleLength >= fileWordsArray.size(0) ) throw new IllegalArgumentException("exampleLength="+exampleLength
				+" cannot exceed number of valid characters in file ("+fileWordsArray.size(0)+")");

		int nRemoved = maxSize - fileWordsArray.size(0);
		System.out.println("Loaded and converted file: " + fileWordsArray.size(0) + " valid characters of "
		+ maxSize + " total characters (" + nRemoved + " removed)");

        initializeOffsets();
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
	public static String[] getMinimalCharacterSet(String textFilePath)throws IOException{
        List<String> wordList = new ArrayList<String>();
        wordList.add("Adjective");
        wordList.add("Numeral");
        wordList.add("Noun");
        wordList.add("Verb");
        wordList.add("PostPositive");
        wordList.add("Punctuation");
        wordList.add("Unknown");
        wordList.add("Conjunction");
        wordList.add("Adverb");
        wordList.add("Determiner");
        wordList.add("Interjection");
        wordList.add("Pronoun");
        wordList.add("Question");
        wordList.add("Duplicator");
      /*  List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),Charset.forName("UTF-8"));
        for( String s : lines ){
            String delimSentence = "[||]";
            String[] sentences = s.split(delimSentence);
            String delims = "[::]";
            if(sentences.length>4) {
                String[] words = sentences[4].split(delims);
                for (String word : words) {
                    if(!word.isEmpty()) {
                        String delim2 = "[ ]";
                        String[] morphemes = word.split(delim2);
                        for (String morpheme : morphemes) {
                            if (!wordList.contains(morpheme))
                                wordList.add(morpheme);
                        }
                    }
                }
            }

        }*/
///
      //  File inputFile = new ClassPathResource("hurriyet_training.txt").getFile();
        List<String> linesSecondFile = Files.readAllLines(new File(textFilePath).toPath(),Charset.forName("UTF-8"));
        for( String s : linesSecondFile ){
            String delims= "[::]";
            String[] words = s.split(delims);
            for (String word : words) {
                if(!word.isEmpty()) {
                    String delim2 = "[ ]";
                    String[] morphemes = word.split(delim2);
                    for (String morpheme : morphemes) {
                        if (!wordList.contains(morpheme))
                            wordList.add(morpheme);
                    }
                }
            }
        }
        ////
      /*  File inputFile2 = new ClassPathResource("test.txt").getFile();
        List<String> linesSecondFile2 = Files.readAllLines(inputFile2.toPath(),Charset.forName("UTF-8"));
        for( String s : linesSecondFile2 ){
            String delims= "[::]";
            String[] words = s.split(delims);
            for (String word : words) {
                if(!word.isEmpty()) {
                    String delim2 = "[ ]";
                    String[] morphemes = word.split(delim2);
                    for (String morpheme : morphemes) {
                        if (!wordList.contains(morpheme))
                            wordList.add(morpheme);
                    }
                }
            }
        }*/
		String[] out = new String[wordList.size()];
		int i=0;
		for( String c : wordList ) {
            if(!(c.compareTo("Unknown")==0)) {
                out[i++] = c;
                System.out.println(c);
            }
        }
		return out;
	}



	public String convertIndexToCharacter( int idx ){
		return validMorphemes[idx];
	}

	public int convertCharacterToIndex( String c ){
		return strToIdxMap.get(c);
	}

	public String getRandomCharacter(){
		return validMorphemes[(int) (rng.nextDouble()*validMorphemes.length)];
	}

	public boolean hasNext() {
		return exampleStartOffsets.size() > 0;
	}
	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
		INDArray input = Nd4j.create(new int[]{currMinibatchSize,validMorphemes.length+14,exampleLength}, 'f');
		INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validMorphemes.length,exampleLength}, 'f');

        INDArray featuresMask = Nd4j.zeros(currMinibatchSize, exampleLength);
        INDArray labelsMask = Nd4j.zeros(currMinibatchSize, exampleLength);
        //int[] temp = new int[2];
        for( int i=0; i<currMinibatchSize; i++ ) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            //int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//Current input
            int currCharIdx = startIdx;
            int c = 0;
            //temp[0] = i;

            for (int j = startIdx + 1; j < endIdx; j++, c++) {
                //temp[1] = j;//BUrada kelime var mı?
                //featuresMask.putScalar(temp, 1.0);
                for (int k = 0; k < fileWordsArray.size(1); k++) {
                    if (((int)fileWordsArray.getDouble(currCharIdx, k)) == 1) {
                        input.putScalar(new int[]{i, k, c}, 1.0);
                    }
                }
                for (int k = 0; k < 14; k++) {
                    if (((int)fileWordsArray.getDouble(currCharIdx+1, k)) == 1) {
                        input.putScalar(new int[]{i,fileWordsArray.size(1)+ k, c}, 1.0);
                    }
                }

                int nextCharIdx = currCharIdx + 1;        //Next character to predict
                for (int k = 14; k < fileWordsArray.size(1); k++) {
                    if (((int)fileWordsArray.getDouble(nextCharIdx, k)) == 1) {
                        labels.putScalar(new int[]{i, k, c}, 1.0);
                    }
                }
                // input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                //labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }
		return new DataSet(input,labels);
	}

	public int totalExamples() {
		return (fileWordsArray.size(0)-1) / miniBatchSize - 2;
	}

	public int inputColumns() {
		return validMorphemes.length;
	}

	public int totalOutcomes() {
		return validMorphemes.length;
	}

	public void reset() {
        exampleStartOffsets.clear();
		initializeOffsets();
	}

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileWordsArray.size(0) - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
    }

	public boolean resetSupported() {
		return true;
	}

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return totalExamples() - exampleStartOffsets.size();
	}

	public int numExamples() {
		return totalExamples();
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}
