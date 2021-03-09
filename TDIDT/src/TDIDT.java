

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

public class TDIDT extends AbstractClassifier implements OptionHandler{

	/** Pour la sérialisation */
	private static final long serialVersionUID = 1L;
	
	/**  Utilisé un nombre d'impurité admissible ?*/
	private boolean m_impurityAllowed = false;
	
	/** Limité la taille de l'arbre ? */
	private boolean m_limitedSize = false;
	
	/** Profondeur maximum de l'abre */
	private int m_treeDepth = -1;
	
	/** Pourcentage d'impureté dans une feuille */
	private int m_impurityRate = 0;
	
	/**Les successeurs du nœud */
	private TDIDT[] m_successors;
	
	/** L'attribut utilisé pour la division */
	private Attribute m_Attribute;
	
	/** Valeur de la classe si le nœud est une feuille */
	private double m_ClassValue;
	
	/** La distribution de la classe si le nœud est une feuille */
	private double[] m_Distribution;
	
	/** L'attribut de la classe de l'ensemble de données */
	private Attribute m_ClassAttribute;
	
	/** Profondeur du nœud */
	private int m_depth;
	
	public TDIDT(int m_depth) {
		m_depth = depth;
	}
	
	@Override
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
	@Override
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
	
	@Override
	public void setOptions(String[] options) throws Exception {
	
		// Options concernant la profondeur
		String treeDepth = Utils.getOption('P', options);
		if(treeDepth.length() != 0) {
			m_treeDepth = Integer.parseInt(treeDepth);
		}else {
			m_treeDepth = -1;
		}
		
		String impurityRate = Utils.getOption('I', options);
		if(impurityRate.length() != 0) {
			m_impurityRate = Integer.parseInt(impurityRate);
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
	@Override
	public String[] getOptions() {
		
		Vector<String> options =new Vector<String>();
		
		if(m_impurityAllowed) {
			options.add("-I");
		}
		if(m_limitedSize) {
			options.add("-P");
		}
		
		Collections.addAll(options, super.getOptions());
		
		return options.toArray(new String[0]);
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		
		// Est-ce que le classifier peut prendre en charge les données?
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
	
	private void makeTree(Instances data) throws Exception {
		
	}

	/**
	 * Classifie une instance de test donnée à l'aide de l'arbre de décision.
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		if(m_Attribute == null) {
			return m_ClassValue;
		}else {
			return m_successors[(int) instance.value(m_Attribute)].
					classifyInstance(instance);
		}
	}

	/**
	 * Calcule la distribution de classe par exemple à l'aide de l'arbre de décision.
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		if(m_Attribute == null) {
			return m_Distribution;
		}else {
			return m_successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
		}
	}
	
	

	@Override
	public String toString() {
		return "TDIDT classifier\n==============\\n" + toString(0);
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
				text.append(m_successors[i].toString(level + 1));
			}
		}
		return text.toString();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
