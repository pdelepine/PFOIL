# TDIDT
TDIDT projet : Construction d'un arbre de décision
				
Le but de ce projet est de réaliser un programme permettant la construction d'un arbre de décision TDIDT
en analysant un jeu de données avec des conditions d'arrêt et en utilisant les classes de Weka.

L'archive est constitué de:
- Du répertoire du projet "TDIDT" contenant :
	- le dossier "src" avec le classificateur DecisionTree.java
	- le .jar de weka 3-8-5
- Un répertoire "Dataset-Valide" contenant des jeux de données sur lesquelles 
le classificateur s'applique. Ces données viennent de https://waikato.github.io/weka-wiki/datasets/

- Le fichier pdf expliquant les choix d'implémentation

- L'éxecutable DecisionTree.jar pour lancer le programme.
						
Pour exécuter le programme, la ligne de commande sur Linux est :
java -jar DecisionTree.jar 

Les options sont : 

-t <nom du fichier d'entrainement> : 
	Définit le fichier d'entrainement. (Venant de weka)

-T <nom du fichier de test > : 
	Définit le fichier de test. S'il est absent, une validation croisée sera effectuée sur les données d'entrainement. (Venant de weka)
	

Les fichiers doit être de type ".arff" et les données des fichiers doivent être nominales et il ne doit pas avoir de données manquantes.

Il existe d'autres options non obligatoire spécifique à notre programme que l'on peut rajouter lors de l'exécution :
					
-P <profondeur maximale> : 
	Définit une profondeur maximale à l'arbre de décision construit
	Valeur par défault: -1 équivalent à aucune 
	Valeurs acceptées: de 1 à infini
	
-I <pourcentage d'impureté> : 
	Définit un taux d'impureté par feuille.
	Valeur par défault: 0
	Valeurs acceptées: de 0 à 100

Par exemple si ce taux est fixé à 5% cela signifie que si le noeud courant contient au moins 95% d'exemples de la même classe,ce noeud devient une feuille de décision.
				
De plus, les options de weka sont aussi disponible, pour les obtenir lancer le programme avec l'option "-h" ou "-help"					
