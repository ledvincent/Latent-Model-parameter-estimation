# Description

Ce dépôt contient une implémentation de l'algorithme Fisher-SGD pour estimer les paramètres d’un modèle de blocs stochastiques (SBM). L'algorithme utilise JAX pour la différenciation automatique et les calculs efficaces. Il permet d’estimer les paramètres des SBM complexes grâce à une descente de gradient stochastique préconditionnée par la matrice d'information de Fisher.
Structure du Dépôt

    algos.py : Contient l'algorithme principal Fisher-SGD pour l'estimation des paramètres, ainsi que les fonctions d'aide pour les mises à jour itératives et la vérification de convergence.

    model.py : Définit la classe du modèle SBM avec des fonctions pour le calcul de la log-vraisemblance, des gradients et l'échantillonnage Gibbs des variables latentes.

    many_estim.py : Génère des données synthétiques SBM et exécute plusieurs estimations pour comparer les résultats.

    plot1run.py : Produit des graphiques pour visualiser l’évolution des paramètres estimés au cours des itérations de Fisher-SGD. Les graphiques sont sauvegardés au format PDF.

## Fonctionnalités Clés

    Algorithme Fisher-SGD :
        Estime les paramètres α (probabilités d’appartenance aux communautés) et π (probabilités de lien entre communautés).
        Utilise le préconditionnement par la matrice d'information de Fisher pour améliorer la convergence.

    Modèle SBM :
        Prend en charge Q communautés (clusters).
        Échantillonne les variables latentes avec un pas de Gibbs sampling.

    Visualisation :
        Génère des graphiques montrant l’évolution des paramètres au cours des itérations.
        Identifie les phases de chauffage et de convergence.

## Prérequis

    Python 3.x
    JAX pour la différenciation automatique et l'accélération GPU
    Matplotlib pour le traçage des graphiques
    Parametrization Cookbook pour la gestion des contraintes des paramètres
    TQDM pour les barres de progression

# Utilisation
1. Générer des Données Synthétiques

Exécutez le script many_estim.py pour générer des données synthétiques SBM et initialiser les paramètres :

python many_estim.py

2. Estimer les Paramètres avec Fisher-SGD

Lancez le script plot1run.py pour effectuer l’estimation des paramètres avec Fisher-SGD et générer les graphiques d'évolution des paramètres :

python plot1run.py

3. Visualiser l’Évolution des Paramètres

Une fois le script plot1run.py exécuté, les graphiques au format PDF seront sauvegardés dans le répertoire courant.
