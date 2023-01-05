# Notes

Notes complétées au fur et à mesure pour bien mémoriser les différents concepts.

# Index des notes
1. [Regression linéaire](#regression-linéaire)
	- [Monovalué : ajustement affine avec une feature](#monovalué-:-ajustement-affine-avec-une-feature)
	    - [Entrainement sur un modele linéaire monovalué](#entrainement-sur-un-modele-linéaire-monovalué)
        - [Fonction coût](#fonction-coût)
        - [Optimisation des paramètres par l'algorithme de la descente de gradient](#optimisation-des-paramètres-par-l'algorithme-de-la-descente-de-gradient)
    - [Algebre linéaire](#algebre-linéaire)

# Regression linéaire

Un modèle de regression linéaire est un modèle qui cherche a établir une relation linéaire entre une variable, dite "expliquée" et une ou plusieurs variables, dites "explicatives".

Dans le cadre de l'apprentissage actuel, ces variables sont souvent nommées :
* x (vecteur de variables), X (matrice de variables), features
* y (vecteur de variables), étiquettes

## Monovalué : ajustement affine avec une feature

Le premier modèle appris dans le cadre du bootcamp ml est l'ajustement affine. Il s'agit de trouver la fonction `f(x) = ax + b`, où `a` et `b` sont des constantes.

Cette fonction est ensuite appelée fonction de prédiction, puisqu'une fois que l'entrainement aura permis d'ajuster les constantes a et b, le modèle permettra de prédire une valeur de y pour un x donné.

### Entrainement sur un modele linéaire monovalué

L'entrainement d'un modèle se fait en plusieurs étapes :
1. Préparation des données : data cleaning (valeurs manquantes, detections des pics avec l'ecart type, ...).
2. Import des données dans le programme et séparation x (features) / y (etiquettes)
3. Si le volume de données le permet, création de deux sets de données : training et test. Les deux sets de données sont crées a partir de la liste totale des données radomisée
4. Entrainement du modèle sur le training set :
    - On sélectionne aléatoirement deux valeurs pour a et b (également appelés θ₁ et θ₀, cette forme permettant ensuite de mieux travailler avec les modèles multivalués). On peut également arbitrairement positionner ces valeurs a 0 ou 1.
    - On fait une prédiction de y pour l'ensemble des valeurs d'entrainement x
    - On calcule le coût global de notre modèle
    - On effectue une optimisation des paramètres de la fonction de prédiction avec l'algorithme de la descente de gradient jusqu'a un certain nombre d'itération ou jusqu'a ce que les paramètres ne varient plus (il convient ici de définir un seuil de variation, souvent appelé "epsilon" en dessous duquel on considère que les paramètres ne varient plus)
    - Les paramètres résultant de l'entrainement sont enregistrés dans le modèle et celui ci est prêt a être testé sur le dataset de test
    - Le modèle établit des prédictions sur le dataset de test avec ses paramètres mis à jour par l'entrainement
    - Les prédictions sont comparées avec les données y du dataset de test et un coût global pour le modèle est établi
    - Le coût est comparé au coût du dataset d'entrainement avec les mêmes paramètres pour déterminer si le modèle est bon sur des données non entrainées (on dit alors qu'il généralise bien) ou si on contraire il n'est pas bon.
    - En fonction du résultat, on peut établir que notre entrainement abouti a un model satisfaisant, ou au contraire a un modèle en overfitting ou en underfitting. Il convient de corriger alors la phase d'entrainement. Différente méthodes existent en fonction de la correction a appliquer.

### Fonction coût

Lors de l'entrainement du modèle, on cherche a réduire le "coût" de nos prédictions sur les données d'entrainement, c'est a dire la somme des écarts entre la valeur de y dans le dataset d'entrainement et la valeur de y prédite grâce a la fonction affine de prédiction.

La somme de ces écarts au carré divisée par 2 fois le nombre d'éléments dans le dataset est appelée "coût".

La "fonction coût" ou "fonction objectif" en optimisation mathématique est la fonction qui établi la relation entre les paramètres de la fonction de prédiction (a et b) et le "coût" global du modèle sur le dataset d'entrainement. En fonction des deux paramètres a et b et projetée dans un espace en 3 dimensions, elle a la forme d'un bol. Lorsqu'on l'observe en fonction d'un seul paramètre (en fixant l'autre) sur un plan en 2 dimensions, elle a la forme d'une courbe quadratique retournée, d'une fonction convexe avec un minimum global qui est la valeur minimum du coût et donc la valeur optimale sur l'axe horizontal du paramètre étudié (ici a ou b).

#### Pourquoi utiliser le carré de la différence dans le calcul du coût ?

La différence entre la prédiction de y et la valeur y du dataset peut être positive ou négative. On peut avoir prédit en dessous ou au dessus de la valeur, mais cela reste une différence.

Pour chaque valeur dans un dataset, la différence peut donc être positive ou négative. Il est alors possible que lorsqu'on fait la somme de ces différences, les valeurs positives et négatives s'annulent, menant alors a une réduction du coût qui ne serait pas représentative de la précision de notre modèle.

Le fait de mettre un nombre au carré permet d'obtenir un nombre positif (- * - = +, + * + = +). Cela va augmenter le coût mais celui ci étant utilisé de manière comparative, la conséquence n'est pas grande. On divise par ailleur la somme des différences au carré par le nombre d'enregistrement du dataset afin de garder la même échelle de valeur peu importe le nombre d'enregistrement.

#### Pourquoi dans la réalité le coût de mon modèle n'est jamais zéro ?

Dans tout système complexe, il y a beaucoup de petites causes indépendantes qui entrent en compte. Ainsi, si je prend l'exemple d'un dataset contenant des informations sur les caractéristiques de maisons et sur leur coût, je ne pourrais pas prédire exactement à coup sur le coût exact pour toutes les maisons car il y a de nombreux paramètres participant à ce coût que je ne suis pas en mesure de maîtriser. Des genre de features incaptables. Mais je pourrais cependant faire une prédiction "correcte" en m'appuyant sur les bons features.

Ces petits "features incaptables" sont appelés le bruit. Selon le théorème central limite, ce bruit se distribue selon une distribution gaussienne.

En quoi le fait d'avoir forcément une partie de l'erreur dûe au bruit gaussien justifie d'utiliser le carré m'échappe encore. **A creuser**.
Bon lien : https://datascience.stackexchange.com/questions/10188/why-do-cost-functions-use-the-square-error


### Optimisation des paramètres par l'algorithme de la descente de gradient

Le but de l'algorithme de la descente de gradient est d'optimiser les paramètres de la fonction de prédiction afin de réduire le coût global du modèle.

La fonction coût associée à un modèle est une fonction convexe dont le minimum global est la valeur optimale sur l'axe horizontal du paramètre étudié.
Notre but est donc d'atteindre cet optimal en prenant pour point de départ sur l'axe vertical le coût de notre modèle avec les paramètres initiaux, et sur l'axe horizontal la valeur initiale du paramètre a optimiser ayant été utilisée pour établir le coût initia.

La descente de gradient fonctionne selon le principe suivant :
1. On calcule pour chaque paramètre (ici a et b) la dérivée partielle de la fonction coût selon ce paramètre. La dérivée ici est la pente de la tagente de la fonction coût en fonction de la valeur du paramètre.
    - La pente est exprimée avec un nombre positif ou négatif plus ou moins élevé. Lorsque la pente est négative, cela signifie que sur le point étudié, la courbe descend. On doit donc déplacer le point vers la droite sur l'axe horizontal pour aller vers le minimum global (et donc augmenter la valeur du paramètre à optimiser). Plus la valeur de la pente est élevée et plus on est loin du minimum global (la courbe s'applati a l'approche de ce minimum). On peut donc ajuster la valeur du paramètre a optimiser de manière conséquente lorsque la dérivée est élevée (en valeur absolue) et plus faiblement lorsque celle ci est faible. La correlation étant forte, on utilise la dérivée (ou en tout cas une partie de la dérivée) pour ajuster le paramètre.
2. Les dérivée partielles de chaque paramètre sont stockées dans un vecteur (au sens informatique). Ce vecteur est appelé gradient.
3. On ajuste les paramètres en meme temps dans le modèle avec une formule simple : `param = param - alpha * dérivée partielle`. Trois choses importantes sur cette formule :
    - On soustrait la dérivée partielle au paramètre puisque lorsque celle ci est négative, on souhaite augmenter la valeur du paramètre et vice-versa lorsque celle ci est positive.
    - On multiplie la dérivée partielle par alpha afin d'ajuster le pas de descente de l'algorithme. En effet, si celui ci est trop grand, l'alogithme peut dépasser le minimum global et avoir un comportement instable. Si celui ci est trop petit, la descente de gradient peut prendre un temps extrèmement long.
    - Il est à noter que la valeur absolue de la dérivée va diminuer au fur et a mesure qu'on se rapproche du minimum global, la pente étant de moins en moins élevée. On va donc naturellement ralentir au fur et à mesure qu'on se rapproche de la valeur optimale du paramètre
4. On réinjecte les valeurs ajustées des paramètres dans le modèle et on boucle sur une nouvelle itération.
5. On arrête d'itérer soit après un nombre défini de boucles, soit lorsque la variation des paramètres entre deux itérations est inférieure a un seuil défini à l'avance (epsilon).

## Algebre linéaire

Nous travaillons avec des vecteurs et des matrices représentant nos jeux de données ainsi que les paramètres du modèle.

Certaines opérations peuvent se faire sur l'ensemble d'une matrice en une seule commande. En l'occurence, l'implémentation de numpy dans Python permet de lancer les opération en multithreading les rendant beaucoup plus rapide que d'itérer sur les matrices pour effectuer des calculs.

Lors des différentes opérations nécessaires à l'entrainement d'un modèle, on peut utiliser l'algebre linéaire. 

- Prédiction : en multipliant le vecteur ou la matrice des features avec le vecteur des paramètres (theta) à l'aide du produit vectoriel, on obtient pour chaque ligne du dataset d'entrainement la somme des feature x paramètre. Pour rendre cela possible, étant donné qu'on a un biais dans la prédiction (le b de f(x) = ax + b) qui ne vient pondérer aucun feature, on ajoute une feature fictive pour chaque enregistrement dont la valeur est 1 (une colonne de 1 dans le vecteur / la matrice des features). Cette colonne est placée en première colonne, et le biais est placé en premier dans le vecteur des paramètres. Ainsi on aura bien le biais multiplié par 1 dans la formule.

- Coût : Lors du calcul du coût on utilise les propriétés de soustractions entre deux vecteurs de même taille pour calculer la différence entre le vecteur de prédiction et le vecteur des étiquettes pour chaque élément, et on fait la somme des carrés du résultat en utilisant produit vectoriel. En effet, faire un produit vectoriel d'un vecteur par lui même revient a faire la somme de ses valeurs au carré.
