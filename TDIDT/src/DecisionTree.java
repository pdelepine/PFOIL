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
	private boolean m_limiteDepth;
	
	/** Booléen pour savoir s'il faut respecter un taux d'impureté par feuille **/
	private boolean m_impurityAllowed;
	
	/** Taux d'impureté par feuille **/
	private double m_impurityRate;
	
	/** Profondeur du noeud courant **/
	private int m_depthTree;
	
	/** Profondeur max de l'arbre **/
	private int m_maxDepthTree;
	
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

	public DecisionTree(int depth, int maxDepth, double impurityRate) {
		m_depthTree = depth;
		m_maxDepthTree = maxDepth;
		m_impurityRate = impurityRate;
	}
	
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
		// class
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
			m_maxDepthTree = Integer.parseInt(treeDepth);
		}else {
			m_maxDepthTree = -1;
		}
		
		String impurityRate = Utils.getOption('I', options);
		if(impurityRate.length() != 0) {
			m_impurityRate = Double.parseDouble(impurityRate);
		}else {
			m_impurityRate = 0;
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
		
		Vector<String> options = new Vector<String>();
		
		if(m_impurityAllowed) {
			options.add("-I");
		}
		if(m_limiteDepth) {
			options.add("-P");
		}
		
		Collections.addAll(options, super.getOptions());
		
		return options.toArray(new String[0]);
	}
	
	public void buildClassifier(Instances instances) throws Exception {
		
		//System.out.println("enterBuild");
		// Est-ce que le classifier peut prendre en charge les données ?
		//getCapabilities().testWithFail(instances);
		if (!instances.classAttribute().isNominal()) {
			throw new Exception("Decision Tree TDIDT: nominal class, please.");
		}
		Enumeration<Attribute> enumAtt = instances.enumerateAttributes();
		while (enumAtt.hasMoreElements()) {
			Attribute attr = (Attribute) enumAtt.nextElement();
			if (!attr.isNominal()) {
				throw new Exception("Decision Tree TDIDT: only nominal attributes, please.");
			}
			Enumeration<Instance>  enumInst = instances.enumerateInstances();
			while (enumInst.hasMoreElements()) {
				if (((Instance) enumInst.nextElement()).isMissing(attr)) {
					throw new Exception("Decision Tree TDIDT: no missing values, please.");
				}
			}
		}

		//System.out.println("endTest");
		
		// On supprime les instances qui n'ont pas de classe
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
			
		if(instances.numAttributes() == 1) {
			System.err.println("On ne peut construire le model (seulement l'attribut "
				+ "de class et présent dans les données");
			return;
		}
				
		makeTree(instances);
		//System.out.println("endMakeTree");
	}
	
	/**
	 * Réalisation de l'arbre de décision TDIDT
	 * @param data
	 * @throws Exception
	 */
	public void makeTree(Instances data) throws Exception {
		
		// Vérifie si aucune instances n'a atteint ce noeud
		if(data.numInstances() == 0) {
			m_Attribute = null;
			m_ClassValue = Utils.missingValue();
			m_Distribution = new double[data.numClasses()];
			return ;
		}
		
		// Calcule l'attribut avec le plus grand gain
		double[] infoGains = new double[data.numAttributes()];
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while(attEnum.hasMoreElements()) {
			Attribute att = attEnum.nextElement();
			infoGains[att.index()] = computeInfoGains(data, att);
		}
		
		m_Attribute = data.attribute(Utils.maxIndex(infoGains));
		
		// Calcule du pourcentage d'impureté sur le noeud
		double[] distriTmp = new double[data.numClasses()];
		Enumeration<Instance> instEnum_ = data.enumerateInstances();
		while(instEnum_.hasMoreElements()) {
			Instance inst = instEnum_.nextElement();
			distriTmp[(int) inst.classValue()] += 1;
		}
		
		// Faire une feuille si le gain est 0 ou si le taux d'impureté est respecté
		// ou on a atteint la taille d'arbre maximum
		// Sinon on crée des successeurs et lance la récursion pour réaliser la suite de l'arbre
		if(Utils.eq(infoGains[m_Attribute.index()], 0)
				|| (distriTmp[Utils.maxIndex(distriTmp)] * 100 / Utils.sum(distriTmp) >= (100 - m_impurityRate))
				|| m_depthTree == m_maxDepthTree ) { 
			m_Attribute = null;
			m_Distribution = new double[data.numClasses()];
			Enumeration<Instance> instEnum = data.enumerateInstances();
			while(instEnum.hasMoreElements()) {
				Instance inst = instEnum.nextElement();
				m_Distribution[(int) inst.classValue()] += 1;
			}
			Utils.normalize(m_Distribution);
			m_ClassValue = Utils.maxIndex(m_Distribution);
			m_ClassAttribute = data.classAttribute();
		}
		else {
			Instances[] splitData = splitData(data, m_Attribute);
			m_Successors = new DecisionTree[m_Attribute.numValues()];
			for(int i = 0; i < m_Attribute.numValues(); ++i) {
				m_Successors[i] = new DecisionTree(m_depthTree+1, m_maxDepthTree, m_impurityRate);
				//System.out.println("Size splitData: "+ splitData[i].size());
				m_Successors[i].buildClassifier(splitData[i]);
				
			}
		}
	}
	
	/**
	 *  Calcul le gain pour un attribut
	 * @param data les données sur lequelles on calcule le gain
	 * @param att L'attribut
	 * @return le gain pour l'attribut donnée
	 */
	public double computeInfoGains(Instances data, Attribute att) throws Exception{
		
		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for(int i = 0; i < att.numValues(); ++i) {
			if(splitData[i].numInstances() > 0) {
				infoGain -= ((double) splitData[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[i]);
			}
		}
		return infoGain;
	}
	
	/**
	 * Calcule de l'entropie d'un jeu de données
	 * @param data
	 * @return
	 */
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
			System.out.println("class null");
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
	
	/**
	 * Divise un jeu de données selon les valeur d'un attribut nominal
	 * @param data le jeu de données
	 * @param att l'attribut nominal
	 * @return le jeu de données divisé
	 */
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
	
	/**
	 * Imprime l'abre de décision selon la méthode toString privée ci-dessous
	 */
	public String toString() {
		return " Decision Tree TDIDT Classifier\n=========================\n" + toString(0);
	}
	/**
	 * Renvoie un arbre depuis un niveau donné
	 * @param level
	 * @return
	 */
	private String toString(int level) {
		
		StringBuffer text = new StringBuffer();
		//System.out.println("test");
		if(m_Attribute == null) {
			//System.out.println("test value: "+ m_ClassValue);
			if(Utils.isMissingValue(m_ClassValue) ) {
				text.append(": null");
			}else {				
				//System.out.print("ClassAttribut null: ");
				//System.out.println(m_ClassAttribute == null);
				if(m_ClassAttribute != null) {
					text.append(": " + m_ClassAttribute.value((int) m_ClassValue) +  "=x" );
				}else {
					text.append("#");
				}
				
			}			
		}else {
			//System.out.print("Size successors:");
			/*if(m_Successors == null) {
				System.out.println("succ null");
			}
			System.out.println((m_Successors == null )?"succ null":m_Successors.length);*/
			for(int i = 0; i < m_Attribute.numValues(); i++) {
				text.append("\n");
				for(int j = 0; j < level; j++) {
					text.append("|  ");
				}
				text.append(m_Attribute.name() + " = " + m_Attribute.value(i));
				
				//System.out.print(i +" m_Successors: ");
				//System.out.println(m_Successors[i] != null);
				if(m_Successors[i] != null) {
					text.append(m_Successors[i].toString(level + 1));
					//System.out.println("test2");
				}
				
			}
		}
		//System.out.println(text);
		return text.toString();
	}

	public static void main(String[] args) {
		try {
			System.out.println(Evaluation.evaluateModel(new DecisionTree(0, 0, 0), args));
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

}
