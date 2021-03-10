import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Instance;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Capabilities.Capability;;

public class DecisionTree extends AbstractClassifier implements OptionHandler {
	
	private static final long serialVersionUID = 1L;
	
	/** Booléen pour savoir s'il faut limité la taille de l'arbre**/
	private boolean m_depthTree;
	
	/** Booléen pour savoir s'il faut respecter un taux d'impureté par feuille **/
	private boolean m_impurityRate;
	
	/** Taux d'impureté par feuille **/
	private double _impurityRate;
	
	/** Profondeur de l'arbre courant **/
	private int _depthTree;
	
	/** Profondeur max de l'arbre **/
	private int _maxDepthTree;
	
	/** Valeur de la classe si le noeud est une feuille **/
	private double m_ClassValue;
	
	/** L'attribut de la classe du jeu de données **/
	private Attribute m_ClassAttribute;
	
	/** La distribution de la classe si le noeud est une feuille **/
	private double[] m_Distribution;
	
	/** L'attribut utilisé pour la séparation des données **/
	private Attribute m_Attribute;
	
	/** Tableau des successeurs du noeud courant **/
	private DecisionTree[] m_Successors;

	public DecisionTree(int depth) {
		_depthTree = depth;
	}
	
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		
		return result;
	}
	
	/**
	 * @return une énumération de toute les options disponibles
	 */
	
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();
		
		newVector.addElement(new Option("\tTaux d'impureté admissile en chaque feuille (0 à 100)\n"
				+ "\tPar défault: 0% ", "I", 1, "-I <taux d'impureté>"));
		newVector.addElement(new Option("\tProfondeur maximum pour l'arbre\n"
				+ "\tPar défault: -1 équivault à aucune", "P", 1, "-P <profondeur>"));
		
		return newVector.elements();
	}
	
	/**
	 * @param les options la liste des options sous forme d'un ensemble de chaînes de caractère
	 * @throws Exception si une option n'est pas supportée
	 */
	
	public void setOptions(String[] options) throws Exception {
	
		// Options concernant la profondeur
		String treeDepth = Utils.getOption('P', options);
		if(treeDepth.length() != 0) {
			_maxDepthTree = Integer.parseInt(treeDepth);
		}else {
			_maxDepthTree = -1;
		}
		
		String impurityRate = Utils.getOption('I', options);
		if(impurityRate.length() != 0) {
			_impurityRate = Double.parseDouble(impurityRate);
		}else {
			_impurityRate = 0;
		}
		
		super.setOptions(options);
		
		Utils.checkForRemainingOptions(options);
	}
	
	/**
	   * Permet d'obtenir les paramètres actuels du classificateur.
	   * 
	   * @return un ensemble de chaînes de caractères pouvant être passées aux setOptions
	   */
	
	public String[] getOptions() {
		
		Vector<String> options =new Vector<String>();
		
		if(m_impurityRate) {
			options.add("-I");
		}
		if(m_depthTree) {
			options.add("-P");
		}
		
		Collections.addAll(options, super.getOptions());
		
		return options.toArray(new String[0]);
	}
	
	public void buildClassifier(Instances instances) throws Exception {
		// Est-ce que le classifier peut prendre en charge les données ?
		getCapabilities().testWithFail(instances);
				
		// On supprime les instances qui n'ont pas de classe
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
			
		if(instances.numAttributes() == 1) {
			System.err.println("On ne peut construire le model (seulement l'attribut "
				+ "de class et présent dans les données");
			return;
		}
				
		makeTree(instances);
	}
	
	// Réalisation de l'arbre de décision
	public void makeTree(Instances data) throws Exception {
		if(data.numInstances() == 0) {
			m_Attribute = null;
			m_Distribution = new double[data.numClasses()];
			return ;
		}
		
		double[] infoGains = new double[data.numAttributes()];
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while(attEnum.hasMoreElements()) {
			Attribute att = attEnum.nextElement();
			infoGains[att.index()] = computeInfoGains(data, att);
		}
		m_Attribute = data.attribute(Utils.maxIndex(infoGains));
		
		if(Utils.eq(infoGains[m_Attribute.index()], _impurityRate) || _depthTree == _maxDepthTree ) { // Alors c'est une feuille
			m_Attribute = null;
			m_Distribution = new double[data.numClasses()];
			Enumeration<Instance> instEnum = data.enumerateInstances();
			while(instEnum.hasMoreElements()) {
				Instance inst = instEnum.nextElement();
				m_Distribution[(int) inst.classValue()]++;
			}
			Utils.normalize(m_Distribution);
			m_ClassValue = Utils.maxIndex(m_Distribution);
			m_ClassAttribute = data.classAttribute();
		}
		else { // On créé un noeud et lance la récursion pour réaliser la suite de l'arbre
			Instances[] splitData = splitData(data, m_Attribute);
			m_Successors = new DecisionTree[m_Attribute.numValues()];
			for(int i = 0; i < m_Attribute.numValues(); ++i) {
				m_Successors[i] = new DecisionTree(_depthTree+1);
				m_Successors[i].buildClassifier(splitData[i]);
			}
		}
	}
	
	// Calcul du gain
	public double computeInfoGains(Instances data, Attribute att) {
		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for(int i = 0; i < att.numValues(); ++i) {
			if(splitData[i].numInstances() > 0) {
				infoGain -= ((double) splitData[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[i]);
			}
		}
		return infoGain;
	}
	
	// Calcul du gain d'entropie
	public double computeEntropy(Instances data) { 
		double[] classCounts = new double[data.numClasses()];
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while(instEnum.hasMoreElements()) {
			Instance inst = instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for(int i = 0; i < data.numClasses(); ++i) {
			if(classCounts[i] > 0) {
				entropy -= classCounts[i] * Utils.log2(classCounts[i]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}
	
	/**
	 * Classifie une instance de test donnée à l'aide de l'arbre de décision.
	 */

	public double classifyInstance(Instance instance) throws Exception {
		if(m_Attribute == null) {
			return m_ClassValue;
		}else {
			return m_Successors[(int) instance.value(m_Attribute)].
					classifyInstance(instance);
		}
	}	
	
	/**
	 * Calcule la distribution de classe par exemple à l'aide de l'arbre de décision.
	 */

	public double[] distributionForInstance(Instance instance) throws Exception {
		if(m_Attribute == null) {
			return m_Distribution;
		}else {
			return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
		}
	}
	
	public Instances[] splitData(Instances data, Attribute att) {
		Instances[] splitData = new Instances[att.numValues()];
		for(int i = 0; i < att.numValues(); ++i) {
			splitData[i] = new Instances(data, data.numInstances());
		}
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while(instEnum.hasMoreElements()) {
			Instance inst = instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		return splitData;
	}
	
	public String toString() {
		return " Decision Tree Classifier \n ================== \n " + toString(0);
	}
	
	private String toString(int level) {
		StringBuffer text = new StringBuffer();
		
		if(m_Attribute == null) {
			text.append(": "+m_ClassAttribute.value((int) m_ClassValue));
		}else {
			for(int i = 0; i < m_Attribute.numValues(); i++) {
				text.append("\n");
				for(int j = 0; j < level; j++) {
					text.append("|  ");
				}
				text.append(m_Attribute.name() + " = " + m_Attribute.value(i));
				text.append(m_Successors[i].toString(level + 1));
			}
		}
		return text.toString();
	}

	public static void main(String[] args) {
		try {
			System.out.println(Evaluation.evaluateModel(new DecisionTree(0), args));
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

}
