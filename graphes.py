"""1 Mots, langages et automates..."""


"""1.1 Mots"""


def pref(mot):
    """Renvoie la liste contenant tous les préfixes du mot donné en paramètre, y compris la chaine vide et le mot lui-même"""
    return [mot[:i] for i in range(len(mot)+1)]


def suf(mot):
    """Renvoie la liste contenant tous les suffixes du mot donné en paramètre, y compris la chaine vide et le mot lui-même"""
    return [mot[i:] for i in range(len(mot)+1)]


def fact(mot):
    """Renvoie la liste sans doublons des facteurs du mot donné en paramètre"""
    return sorted(list(set(mot[i:j] for i in range(len(mot)+1) for j in range(i, len(mot)+1))))


def miroir(mot):
    """Renvoie le mot miroir de celui passé en paramètre"""
    return mot[::-1]


"""1.2 Langages"""


def concatene(L1, L2):
    """Renvoie le produit de concaténation sans doublons des deux langages passés en paramètres"""
    return sorted(list({x + y for x in L1 for y in L2}))


def puis(L1, puiss):
    """Renvoie le langage L^n sans doublons"""
    Ln = {''}  # le langage L^0 est le singleton {''} (chaîne vide)
    for _ in range(puiss):
        # on ajoute toutes les lettres de L à la fin des mots de L^(i-1)
        Ln = {mot + lettre for mot in Ln for lettre in L1}
    return sorted(list(Ln))


def tousmots(L1, n):
    """Renvoie la liste de tous les mots de A* de longueur inférieure à n, étant donné un alphabet passé en paramètre"""
    L2 = ['']
    for i in range(1, n+1):
        # augmente la puissance de L^n jusqu'à n
        L2 += puis(L1, i)
    return L2


def defauto():
    """Permet de faire la saisie d'un automate sans doublon"""
    alphabet = input(
        "Entrez l'alphabet, avec les différents symboles séparés par des virgules : ").split(",")
    etats = convertint(input("Entrez les états, séparés par des virgules : "))
    transitions = []
    print("Apuuyez sur entrée pour valider")
    while True:  # tant que l'utilisateur rentre des transitions
        trans = input(
            "Entrez l'état de départ, le symbole de transition et l'état d'arrivée séparés par des virgules : ")
        if not trans:  # si l'utilisateur appuie sur entree sans rien écrire
            break
        trans = trans.split(",")  # séparer les éléments de la transition
        # convertir les états en int
        trans[0] = int(trans[0])
        trans[2] = int(trans[2])
        if trans not in transitions and trans[0] in etats and trans[1] in alphabet and trans[2] in etats:
            # si la transition n'est pas déjà dans la liste
            # et que les états de la transition sont bien dans les états de l'automate
            # et que le symbole de transition est bien dans l'alphabet de l'automate
            transitions.append(trans)
    initiaux = convertint(
        input("Entrez les états initiaux, séparés par des espaces : "))
    finaux = convertint(
        input("Entrez les états finaux, séparés par des espaces : "))
    return {"alphabet": alphabet, "etats": etats, "transitions": transitions, "I": initiaux, "F": finaux}


def convertint(chaine):
    """Convertit une liste de chaines de caractères séparées par des virgules en liste d'int"""
    converted = set()
    for element in chaine.split(","):
        # on convertit chaque élément (séparé par une virgule) en int et on l'ajoute à l'ensemble
        converted.add(int(element))
    return list(converted)


def lirelettre(T, E, a):
    """Etant donnés en paramètres une liste de transitions T, une liste d'états E et une lettre a,
    Renvoie la liste des états dans lesquels on peut arriver en partant d'un état de E et en lisant la lettre a"""
    etatspossible = []
    for transition in T:
        # si la transition part bien de E et lit a, et que l'état d'arrivée n'est pas encore dans la liste, alors on l'ajoute
        if transition[0] in E and transition[1] == a and transition[2] not in etatspossible:
            etatspossible.append(transition[2])
    return etatspossible


def liremot(T, E, m):
    """Etant donnés en paramètres une liste de transitions T, une liste d'états E et un mot m,
    Renvoie la liste des états dans lesquels on peut arriver en partant d'un état de E et en lisant le mot m"""
    etatspossible = set(E)
    for symbole in m:  # pour toutes les lettres du mot m
        # on ajoute à l'ensemble d'états possibles ceux qui sont atteignables par les symboles de m
        etatspossible = set(lirelettre(T, etatspossible, symbole))
    return list(etatspossible)


def accepte(auto, m):
    """Prend en paramètres un automate et un mot m et renvoie True si le mot m est accepté par l'automate"""
    # Renvoie True si au moins un chemin de l'automate partant d'un état initial puis en lisant m mène à un état final
    return any(etat in auto['F'] for etat in liremot(auto["transitions"], auto["I"], m))


def langage_accept(auto, n):
    """Prend en paramètres un automate et un entier n et renvoie la liste des mots de longueur inférieure à n acceptés par l'automate"""
    dico_mot = dict()  # dictionnaire qui contient les mots de longueur i
    dico_mot[0] = {""}
    mots_possibles = []
    for i in range(1, n+1):
        dico_mot[i] = set()
        for symbole in auto['alphabet']:
            # pour toutes les lettres de l'alphabet
            for prefixe in dico_mot[i-1]:
                # pour toutes les suites de lettres faisables grace à l'automate

                # on ajoute la concaténation des lettres avec celle de l'alphabet aux suites de lettres faisables
                dico_mot[i].add(prefixe+symbole)

                if accepte(auto, prefixe+symbole):
                    # si le langage accepte ce mot, alors on l'ajoute aux mots acceptés
                    mots_possibles.append(prefixe+symbole)
    return mots_possibles


def deterministe(auto):
    """Etant donné un automate passé en paramètre renvoie True s'il est déterministe et False sinon"""
    # si l'automate a plus d'un état initial alors il n'est pas déterministe
    if len(auto['I']) > 1:
        return False
    transi_set = set()  # ensemble des transitions
    for transition in auto["transitions"]:
        # ajout d'un tuple (état de départ, symbole de transition) à l'ensemble
        transi_set.add((transition[0], transition[1]))
    # si le nombre de transitions est différent du nombre d'éléments de l'ensemble alors il y a des doublons
    return len(transi_set) == len(auto["transitions"])


def determinise(auto):
    """Déterminise l'automate passé en paramètre et renvoie l'automate déterministe"""
    # on initialise la liste des états de l'automate déterministe avec l'état initial
    auto["etats"] = [auto['I']]
    # on initialise la liste des transitions de l'automate déterministe et ajoutons les autres états
    auto["transitions"] = determinise_aux([], auto, auto['I'])
    # on rassemble les états initiaux en un seul état
    auto['I'] = [auto['I']]
    # on remplace les états finaux par les états qui contiennent un état final
    auto['F'] = [e for e in auto["etats"] if any(f in auto['F'] for f in e)]
    return auto


def determinise_aux(transition, auto, etats):
    """Etant donnés en paramètres une liste de transitions, un automate et une liste d'états, 
    Renvoie la liste des transitions de l'automate déterministe"""
    if etats == []:
        return transition
    for symbole in auto["alphabet"]:
        # pour chaque symbole de l'alphabet on regarde les états possibles à partir de l'état courant et en lisant le symbole
        etatspossibles = lirelettre(auto["transitions"], etats, symbole)
        if etatspossibles != [] and ([etats, symbole, etatspossibles] not in transition):
            # si on a des états possibles et que la transition n'est pas déjà dans la liste des transitions alors on l'ajoute
            transition.append([etats, symbole, etatspossibles])
            if etatspossibles not in auto["etats"]:
                # on ajoute les états possibles à la liste des états de l'automate déterministe
                auto["etats"].append(etatspossibles)
            # on ajoute les transitions possibles à partir des états possibles
            determinise_aux(transition, auto, etatspossibles)
    return transition


def renommage(auto):
    """Etant donné un automate passé en paramètre, renomme ses états avec les premiers entiers et renvoie l'automate"""
    renom = dict()
    etats = auto["etats"]  # liste des états de l'automate
    for i in range(len(etats)):
        renom[str(etats[i])] = i  # on associe à chaque état un entier
    # on renomme les états, les états initiaux, les états finaux et les transitions
    auto["etats"] = renommage_aux(auto["etats"], renom)
    auto['I'] = renommage_aux(auto["I"], renom)
    auto['F'] = renommage_aux(auto["F"], renom)
    auto["transitions"] = renommage_transi(auto["transitions"], renom)
    return auto


def renommage_aux(ensemble, renom):
    """Etant donnés en paramètres un ensemble d'états et un dictionnaire de renommage, renvoie la liste des états renommés"""
    if (ensemble==[]):
        return []
    if type(ensemble) != list or type(ensemble[0]) == int:
        # si l'ensemble n'est pas une liste on le transforme en liste
        ensemble = [ensemble]
    # on renomme les états de l'ensemble à l'aide du dictionnaire et des entiers associés
    return [renom.get(str(etat)) for etat in ensemble]


def renommage_transi(ensemble, renom):
    """Etant donnés en paramètres un ensemble de transitions et un dictionnaire de renommage, renvoie la liste des transitions renommées"""
    return [[renom.get(str(ensemble[i][0])), ensemble[i][1], renom.get(str(ensemble[i][2]))] for i in range(len(ensemble))]  # on renomme les états de la transition à l'aide du dictionnaire et des entiers associés


def set_transitions(auto):
    """Etant donné un automate passé en paramètre renvoie l'ensemble des transitions sous la forme d'un tuple (état de départ, symbole de transition)"""
    return {(transition[0], transition[1]) for transition in auto["transitions"]}


def complet(auto):
    """Etant donné un automate passé en paramètre renvoie True s'il est complet et False sinon"""
    transitions = set_transitions(
        auto)  # ensemble des transitions sous la forme d'un tuple (état de départ, symbole de transition)
    for etat in auto["etats"]:
        for symbole in auto["alphabet"]:
            if (etat, symbole) not in transitions:
                # si une transition n'est pas dans l'ensemble des transitions pour chaque état et chaque lettre,
                # alors l'automate n'est pas complet
                return False
    return True


def complete(auto):
    """Complète l'automate passé en paramètre en ajoutant un état puits"""
    if complet(auto):
        # si l'automate est complet on le renvoie
        return auto
    # le puit est l'état suivant le dernier état de l'automate
    puit = len(auto["etats"])
    auto["etats"].append(puit)  # on ajoute l'état puits à la liste des états
    transitions = set_transitions(auto)
    for etat in auto["etats"]:
        for symbole in auto["alphabet"]:
            if (etat, symbole) not in transitions:
                # si une transition n'est pas dans l'ensemble des transitions pour chaque état et chaque lettre,
                # alors on ajoute une transition vers l'état puits
                auto["transitions"].append([etat, symbole, puit])
    return auto


def complement(auto):
    """Etant donné un automate passé en paramètre acceptant un langage L, renvoie un automate acceptant le complement L¯."""
    auto = complete(renommage(determinise(auto))
                    )  # on détermine, on renomme et on complète l'automate
    # on inverse les états finaux et non finaux
    auto['F'] = [etat for etat in auto["etats"] if etat not in auto['F']]
    return auto


def produit(auto1, auto2):
    """Etant donnés deux automates déterministes passés en paramètres acceptant respectivement les langages L1 et L2,
    Renvoie l'automate produit acceptant le langage L1 x L2"""
    # on rassemble les états initiaux des deux automates en un seul état
    initiaux = (auto1['I'][0], auto2['I'][0])
    # on initialise la liste des états de l'automate avec les états initiaux des deux automates
    auto_etats = [initiaux]
    # on initialise la liste des transitions de l'automate avec les transitions possibles à partir des états initiaux et ajoutons les autres états
    transitions = produit_aux(auto1, auto2, initiaux, [], auto_etats)
    return {"alphabet": list(sorted(set(auto1["alphabet"]+auto2["alphabet"]))), "etats": auto_etats, "transitions": transitions, "I": initiaux}


def produit_aux(auto1, auto2, etats, transitions, auto_etats):
    """Etant donnés en paramètres deux automates déterministes, une liste d'états, une liste de transitions et une liste d'états de l'automate produit,
    Renvoie la liste des transitions de l'automate produit acceptant le langage L1 x L2"""
    for symbole in set(auto1["alphabet"]+auto2["alphabet"]):
        # pour chaque symbole des deux alphabets confondus, on regarde les états possibles à partir de l'état courant
        etatspossibles1 = lirelettre(auto1["transitions"], [etats[0]], symbole)
        etatspossibles2 = lirelettre(auto2["transitions"], [etats[1]], symbole)

        if etatspossibles1 != [] and etatspossibles2 != []:
            # si les deux listes ne sont pas vides, on crée un nouvel état possible pour l'automate produit
            for etatspossibles11 in etatspossibles1:
                for etatspossibles22 in etatspossibles2:
                    etatspossibles = (etatspossibles11, etatspossibles22)
                    if ([etats, symbole, etatspossibles] not in transitions):
                        # si la transition n'est pas déjà dans la liste des transitions, on l'ajoute
                        transitions.append([etats, symbole, etatspossibles])
                        if etatspossibles not in auto_etats:
                            # si l'état n'est pas déjà dans la liste des états, on l'ajoute
                            auto_etats.append(etatspossibles)
                        # on ajoute les transitions possibles à partir des états possibles
                        produit_aux(auto1, auto2, etatspossibles,
                                    transitions, auto_etats)
    return transitions


def inter(auto1, auto2):
    """Etant donnés deux automates déterministes passés en paramètres acceptant respectivement les langages L1 et L2,
    Renvoie l'automate produit acceptant l'intersection L1 ∩ L2"""
    auto_inter = produit(auto1, auto2)  # on fait le produit des automates
    # les états finaux de l'automate sont ceux qui contiennent ceux finaux de l'automate 1 et ceux finaux de l'automate 2   
    auto_inter['F'] = [etat for etat in auto_inter["etats"]
                       if etat[0] in auto1['F'] and etat[1] in auto2['F']]
    return auto_inter


def difference(auto1, auto2):
    """Etant donnés deux automates déterministes passés en paramètres acceptant respectivement les langages L1 et L2,
    Renvoie l'automate produit acceptant la différence L1 \ L2"""
    # on complète les automates
    complete(auto1)
    complete(auto2)
    auto_diff = produit(auto1, auto2)  # on fait le produit des automates
    # les états finaux de l'automate sont ceux qui contiennent ceux finaux de l'automate 1, sans les finaux de l'automate 2
    auto_diff['F'] = [etat for etat in auto_diff["etats"]
                      if etat[0] in auto1['F'] and etat[1] not in auto2['F']]
    return auto_diff


def prefixe(auto):
    """Etant donné un automate émondé acceptant L,
    Renvoie un automate acceptant l'ensemble des préfixes des mots de L"""
    nouveauto = dict(auto)  # on crée une copie de l'automate
    # les états finaux sont tous les états de l'automate
    nouveauto['F'] = nouveauto["etats"]
    return nouveauto


def suffixe(auto):
    """Etant donné un automate émondé acceptant L,
    Renvoie un automate acceptant l'ensemble des suffixes des mots de L"""
    nouveauto = dict(auto)  # on crée une copie de l'automate
    # les états initiaux sont tous les états de l'automate
    nouveauto['I'] = nouveauto["etats"]
    return nouveauto


def facteur(auto):
    """Etant donné un automate émondé acceptant L,
    Renvoie un automate acceptant l'ensemble des facteurs des mots de L"""
    nouveauto = dict(auto)  # on crée une copie de l'automate
    # les états initiaux sont tous les états de l'automate
    nouveauto['I'] = nouveauto["etats"]
    # les états finaux sont tous les états de l'automate
    nouveauto['F'] = nouveauto["etats"]
    return nouveauto


def miroir_graphes(auto):
    """Etant donné un automate émondé acceptant L,
    Renvoie un automate acceptant l'ensemble des miroirs des mots de L"""
    nouveauto = dict(auto)  # on crée une copie de l'automate
    # on inverse les transitions
    nouveauto["transitions"] = [transition[::-1]
                                for transition in auto["transitions"]]
    # les états initiaux sont les états finaux de l'automate
    nouveauto['I'] = auto['F']
    # les états finaux sont les états initiaux de l'automate
    nouveauto['F'] = auto['I']
    return nouveauto


def minimise(auto):
    """Etant donné un automate complet et déterministe,
    Renvoie l'automate minimisé"""
    classes = list()
    classes.append(set(auto["F"]))  # on ajoute dans la classe les états finaux
    # on ajoute dans la classe les états non finaux
    classes.append({etat for etat in auto["etats"] if etat not in auto["F"]})
    # on crée les groupes d'états équivalents
    auto["etats"] = [list(s) for s in classes_etats(
        auto, classes, dico_transi(auto))] # transforme en liste les ensembles d'états équivalents

    dico_classe = dico_classes(auto["etats"]) # dictionnaire associant à chaque état, son nouvel état selon sa classe

    finaux = []
    for etat in auto["F"]:
        if dico_classe[etat] not in finaux:
            # pour chaque état final, on récupère son nouvel état et on l'ajoute à la liste des finaux s'il n'y est pas déjà
            finaux.append(dico_classe[etat])
    auto['F'] = finaux # on associe la liste des finaux créée
    # on récupère les états initiaux en accédant à la valeur du dictionnaire associée à la clé de l'état initial de l'automate
    initiaux = dico_classe[auto['I'][0]] 
    auto['I'] = list(initiaux) # on met les états initiaux en liste
    # on minimise les transitions
    auto["transitions"] = chemin_minimise(auto, [], initiaux, dico_classe)
    return auto


def classes_etats(auto, classes, dico_transi):
    """Etant donnés un automate, une liste de classes d'équivalence et un dictionnaire de transitions,
    Renvoie la liste des classes d'équivalence"""
    new_classes = []
    auto_etats = auto["etats"]
    for i in range(len(auto_etats)):
        # pour chaque état, on crée une classe d'équivalence contenant cet état
        classe = set()
        classe.add(auto_etats[i])
        for j in range(len(auto_etats)):
            # on ajoute dans la classe tous les états qui sont dans la même classe d'équivalence
            if (meme_classe(auto_etats[i], auto_etats[j], dico_transi, classes, auto)):
                classe.add(auto_etats[j])
        if classe not in new_classes:
            # on ajoute la classe à la liste des classes d'équivalence si elle n'y est pas déjà
            new_classes.append(classe)

    if classes != new_classes:  # si les classes ont changé, on continue le processus
        return classes_etats(auto, new_classes, dico_transi)
    return classes


def meme_classe(etat1, etat2, dico_transi, classes, auto):
    """Etant donnés deux états, un dictionnaire de transitions, une liste de classes d'équivalence et un automate,
    Renvoie True si les deux états sont dans la même classe d'équivalence, False sinon"""
    for symbole in auto["alphabet"]:  # pour chaque symbole de l'alphabet
        for set in classes:  # pour chaque classe d'équivalence
            # si les deux états ne sont pas dans la même classe d'équivalence, on renvoie False
            if (not (etat1 in set and etat2 in set) and (etat1 in set or etat2 in set)):
                return False
            etat1inset = dico_transi[(etat1, symbole)] in set
            etat2inset = dico_transi[(etat2, symbole)] in set
            if (not (etat1inset and etat2inset) and (etat1inset or etat2inset)):
                return False
    return True


def chemin_minimise(auto, transitions, etats, dico_classes):
    """Etant donnés un automate, une liste de transitions, une liste d'états et un dictionnaire associant à chaque état, son nouvel état,
    Renvoie la liste des transitions minimisées"""
    for symbole in auto["alphabet"]:
        # pour chaque symbole de l'alphabet, on récupère les états possibles
        etatspossibles = lirelettre(auto["transitions"], [etats[0]], symbole)
        if etatspossibles != []:
            # si il y a des états possibles, on récupère la classe de l'état possible
            classepossible = dico_classes[etatspossibles[0]]
            if ([etats, symbole, classepossible] not in transitions):
                # si la transition n'est pas déjà dans la liste des transitions, on l'ajoute
                transitions.append([etats, symbole, classepossible])
                # on continue le processus avec les états possibles
                chemin_minimise(auto, transitions,
                                classepossible, dico_classes)
    return transitions


def dico_classes(etats_minimise):
    """Etant donné une liste de classes d'états,
    Renvoie un dictionnaire associant à chaque état son nouvel état"""
    dico_classes = dict()  # dictionnaire associant à chaque état, son nouvel état
    for i in range(len(etats_minimise)):  # pour chaque groupe d'états dans la classe d'équivalence
        for etat in etats_minimise[i]:  # pour chaque état de ce groupe
            dico_classes[etat] = etats_minimise[i] # on associe l'état à son nouvel état
    return dico_classes


def dico_transi(auto):
    """Etant donné un automate,
    Renovie un dictionnaire associant à chaque transition son état d'arrivée"""
    dico_transi = dict()
    for transition in auto["transitions"]: # pour chaque transition
        # on associe la transition à son état d'arrivée
        # sous la forme (état de départ, symbole) : état d'arrivée
        dico_transi[(transition[0], transition[1])] = transition[2] 
    return dico_transi


if __name__ == '__main__':
    print(pref("coucou"))
    print(suf("coucou"))
    print(fact("coucou"))
    print(miroir("coucou"))

    L1 = ['aa', 'ab', 'ba', 'bb']
    L2 = ['a', 'b', '']
    print(concatene(L1, L2))
    print(sorted(['aaa', 'aab', 'aa', 'aba', 'abb',
          'ab', 'baa', 'bab', 'ba', 'bba', 'bbb', 'bb']))

    L1 = ['aa', 'ab', 'ba', 'bb']
    print(puis(L1, 2))
    print(['aaaa', 'aaab', 'aaba', 'aabb', 'abaa', 'abab', 'abba', 'abbb',
          'baaa', 'baab', 'baba', 'babb', 'bbaa', 'bbab', 'bbba', 'bbbb'])

    """1.2.3. Pourquoi ne peut-on pas faire de fonction calculant l'étoile d'un langage ?

    On ne peut pas faire de fonction calculant l'étoile d'un langage car cette fonction devra générer une liste infini, ce qui est impossible.
    """
    print(tousmots(['a', 'b'], 3))
    ['a', 'b', 'aa', 'ab', 'ba', 'bb', 'aaa', 'aab',
        'aba', 'abb', 'baa', 'bab', 'bba', 'bbb', '']

    #auto = defauto()
    #print(auto)

    auto = {"alphabet": ['a', 'b'], "etats": [1, 2, 3, 4],
            "transitions": [[1, 'a', 2], [2, 'a', 2], [2, 'b', 3], [3, 'a', 4]],
            "I": [1], "F": [4]}
    print(lirelettre(auto["transitions"], auto["etats"], 'a'))
    print(liremot(auto["transitions"], auto["etats"], 'aba'))

    print(accepte(auto, "aba"))
    print(accepte(auto, ""))

    print(langage_accept(auto, 3))

    """1.3.6. Pourquoi ne peut-on pas faire une fonction qui renvoie le langage accepté par un automate ?

    Un langage peut boucler et contenir des mots de longueur infinie et donc il n'est pas possible de renvoyer une liste de tous les mots acceptés par l'automate.

    """
    auto0 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2, 3], "transitions": [
        [0, 'a', 1], [1, 'a', 1], [1, 'b', 2], [2, 'a', 3]], "I": [0], "F": [3]}
    auto1 = {"alphabet": ['a', 'b'], "etats": [0, 1], "transitions": [
        [0, 'a', 0], [0, 'b', 1], [1, 'b', 1], [1, 'a', 1]], "I": [0], "F": [1]}
    auto2 = {"alphabet": ['a', 'b'], "etats": [0, 1], "transitions": [
        [0, 'a', 0], [0, 'a', 1], [1, 'b', 1], [1, 'a', 1]], "I": [0], "F": [1]}

    print(deterministe(auto0))
    print(deterministe(auto2))
    print(determinise(auto2))

    print(renommage(determinise(auto2)))

    auto3 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2], "transitions": [
        [0, 'a', 1], [0, 'a', 0], [1, 'b', 2], [1, 'b', 1]], "I": [0], "F": [2]}
    print(complet(auto0))
    print(complet(auto1))

    print(complete(auto0))

    print(complement(auto3))

    auto4 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2,], "transitions": [
        [0, 'a', 1], [1, 'b', 2], [2, 'b', 2], [2, 'a', 2]], "I": [0], "F": [2]}
    auto5 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2], "transitions": [[0, 'a', 0], [
        0, 'b', 1], [1, 'a', 1], [1, 'b', 2], [2, 'a', 2], [2, 'b', 0]], "I": [0], "F": [0, 1]}

    print(inter(auto4, auto5))
    print(renommage(inter(auto4, auto5)))

    print(difference(auto4, auto5))
    print(renommage(difference(auto4, auto5)))

    auto_fermeture = {"alphabet": ['a', 'b'], "etats": [1, 2, 3, 4, 5], "transitions": [[1, 'a', 1], [
        1, 'a', 2], [2, 'a', 5], [2, 'b', 3], [3, 'b', 3], [3, 'a', 4], [5, 'b', 5]], "I": [1], "F": [4, 5]}

    print(miroir_graphes(auto_fermeture))

    auto6 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2, 3, 4, 5], "transitions": [[0, 'a', 4], [0, 'b', 3], [1, 'a', 5], [1, 'b', 5], [
        2, 'a', 5], [2, 'b', 2], [3, 'a', 1], [3, 'b', 0], [4, 'a', 1], [4, 'b', 2], [5, 'a', 2], [5, 'b', 5]], "I": [0], "F": [0, 1, 2, 5]}
    print(renommage(minimise(auto6)))

    mini = {"alphabet": ['a', 'b'], "etats": [1, 2, 3, 4, 5, 6], "transitions": [[1, 'a', 2], [1, 'b', 4], [2, 'a', 3], [2, 'b', 4], [
        3, 'b', 4], [3, 'a', 3], [4, 'a', 5], [4, 'b', 6], [5, 'a', 5], [5, 'b', 5], [6, 'a', 5], [6, 'b', 6]], "I": [1], "F": [5, 6]}
    

    autotest = {"alphabet": ['a', 'b'], "etats": [1, 2, 3, 4, 5], "transitions": [[1, 'a', 2], [2, 'b', 3], [
        3, 'b', 4], [3, 'a', 1], [5, 'a', 2], [5, 'b', 3]], "I": [1], "F": [1]}
    #print(determinise(autotest))

    autointer1 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2], "transitions": [[0, 'a', 1], [0, 'b', 2], [1, 'a', 2], [1, 'b', 2], [
        2, 'a', 2], [2, 'b', 1]], "I": [0], "F": [0, 2]}
    
    autointer2 = {"alphabet": ['a', 'b'], "etats": [0, 1, 2], "transitions": [[0, 'a', 1], [0, 'b', 0], [1, 'a', 2], [1, 'b', 0], [
        2, 'a', 0], [2, 'b', 1]], "I": [0], "F": [1]}
    
    #print(difference(autointer1, autointer2))

    partie5 = {"alphabet": ['a', 'b'], "etats": [1, 2, 3, 4, 5], "transitions": [[1, 'a', 1], [1, 'a', 2], [
        2, 'a', 5], [2, 'b', 3], [3, 'a', 4], [3, 'b', 3], [5, 'b', 5]], "I": [1], "F": [4, 5]}
    
    #print("prefixe\n",prefixe(partie5))
    #print("suffixe\n",suffixe(partie5))
    #print("facteur\n",facteur(partie5))
    #print("miroir_graphes\n",miroir_graphes(partie5))

    auto_determ = {
        "alphabet": ['a', 'b'],
        "etats": [1, 2, 3, 4],
        "transitions": [[1, 'a', 2], [1, 'b', 2], [2, 'b', 1], [2, 'a', 4], [4, 'a', 2], [4, 'b', 3], [3, 'b', 4], [1, 'a', 3]],
        "I": [1, 4],
        "F": [3, 4]
    }

    print(determinise(auto_determ))
    
