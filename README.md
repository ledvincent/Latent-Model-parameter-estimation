# Description

Ce dépôt contient deux notebooks d'implémentation de l'algorithme Fisher-SGD pour estimer les paramètres d’un modèle de mélange gaussien (2 lois gaussiennes) et un modèle à effet mixte. Les gradients sont calculés analytiquement.

**Fisher Stochastic Gradient Descent** (Fisher SGD) est une variante de l'algorithme de Descente de Gradient Stochastique (SGD) utilisée pour l'estimation des paramètres. La principale innovation de Fisher SGD réside dans l'intégration du préconditionnement basé sur l'information de Fisher, une mesure de la quantité d'information qu'un observateur peut obtenir sur les paramètres du modèle à partir des données observées.

### Chaque notebook guide à travers les étapes suivantes :

    Simulation des Données : Génération de données synthétiques avec des paramètres connus.
    Définition du Modèle : Implémentation des fonctions de log-vraisemblance et de gradient.
    Échantillonnage des Variables Latentes : Utilisation de méthodes MCMC (Gibbs Sampling) pour les modèles avec variables latentes.
    Estimation des Paramètres : Application de la Descente de Gradient Stochastique avec préconditionnement par l'information de Fisher.
    Visualisation des Résultats : Suivi de la convergence des paramètres au cours des itérations.

Veuillez lancer le code un par un.

### Code Python qui utilise uniquement les modules: **tqdm** et **numpy**

