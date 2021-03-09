

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;

public class TDIDT extends AbstractClassifier implements OptionHandler{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/**  Utilisé un nombre d'impurité admissible ?*/
	private boolean m_impurityRate = false;
	/** Limité la taille de l'arbre */
	private boolean m_limitedSize = false;
	
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		
		// class
		result.enable(Capability.NOMINAL_CLASS);
		
		return result;
	}	
	
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();
		
		newVector.addElement(new Option("\tTaux d'impureté admissile en chaque feuille\n"
				+ "\tPar défault: 0%", "I", 1, "-I <taux d'impureté"));
		newVector.addElement(new Option("\tProfondeur maximum pour l'arbre\n"
				+ "\tPar défault: aucun", "P", 1, "-P <profondeur>"));
		
		return newVector.elements();
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		super.setOptions(options);
	}
	
	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return super.getOptions();
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
