import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;;

public class DecisionTree extends AbstractClassifier implements OptionHandler {
	
	private static final long serialVersionUID = 1L;
	
	private boolean m_depthTree;
	private boolean m_impurityRate;
	private double _impurityRate;
	private int _depthTree;
	private int _maxDepthTree;
	private double m_ClassValue;
	private Attribute m_ClassAttribute;
	private double[] m_Distribution;
	private Attribute m_Attribute;
	private DecisionTree[] m_Successors;

	public DecisionTree(int depth) {
		_depthTree = depth;
	}
	
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NOMINAL_CLASS);
		
		return result;
	}
	
	public void buildClassifier(Instances data) throws Exception {
		if(data.classAttribute().isNominal()) {
			
		}
		else {
			throw new Exception("Nominal class needed");
		}
	}
	
	public void makeTree(Instances data) throws Exception {
		if(data.numInstances() == 0) {
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
		else {
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
		return " Decision Tree Classifier \n ============== \n ";
	}

	public static void main(String[] args) {
		
	}

}
