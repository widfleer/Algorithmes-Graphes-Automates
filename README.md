# Exploration algorithmique d'un problème

Ce projet contient des fonctions permettant de manipuler des mots, langages et automates finis.

## Fonctions sur les mots

- `pref`, `suf`, `fact`, `miroir` : calculent respectivement les préfixes, suffixes, facteurs et miroir d'un mot

## Fonctions sur les langages 

- `concatene`, `puis` : calculent respectivement le produit de concaténation et la puissance d'un langage  
- `tousmots` : génère tous les mots possibles sur un alphabet jusqu'à une longueur donnée
- `defauto` : permet de définir un automate en entrée

## Manipulation d'automates

Les fonctions suivantes permettent de :

- Lire un mot/lettre sur un automate
- Vérifier qu'un mot est accepté
- Calculer le langage accepté 
- Vérifier la déterminisation, complétion, etc.
- Calculer des opérations sur les automates (produit, complement, intersection, difference)
- Manipuler les états, transitions, déterminiser, minimiser, etc.

## Fermetures des langages

Fonctions calculant les fermetures des langages acceptés par des automates :

- Préfixes
- Suffixes  
- Facteurs
- Miroirs

## Exemples

Enfin, le code contient plusieurs exemples d'utilisation de ces fonctions sur des automates simples.
