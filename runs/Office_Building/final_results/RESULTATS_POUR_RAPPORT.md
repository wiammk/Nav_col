# Résultats expérimentaux à utiliser dans le rapport

## Protocole

Les résultats reposent sur dix graines d'entraînement indépendantes (42 à 51), quatre tailles de flotte (1, 3, 5 et 10 robots) et 200 scénarios de test gelés par configuration. Les politiques ne sont pas mises à jour pendant le test. Les intervalles de confiance à 95 % sont calculés entre les graines avec la loi de Student. Les comparaisons utilisent un test exact de permutation des signes sur les résultats appariés par graine. La correction de Holm est appliquée séparément à chaque comparaison planifiée et à chaque taille de flotte, avec le taux de succès et le SPL comme critères principaux.

## Comparaison des architectures Q-learning

| Robots | Local indépendant | Centralisé | FedAvg Q-learning |
|---:|---:|---:|---:|
| 1 | 86,40 ± 1,62 % | 86,40 ± 1,62 % | 86,40 ± 1,62 % |
| 3 | 64,92 ± 1,22 % | 70,93 ± 1,13 % | **73,48 ± 1,25 %** |
| 5 | 49,95 ± 1,54 % | 55,02 ± 0,80 % | **55,85 ± 1,10 %** |
| 10 | 34,47 ± 0,23 % | **41,94 ± 0,61 %** | 41,02 ± 0,58 % |

Avec un seul robot, les trois architectures sont identiques, ce qui est cohérent puisque l'apprentissage fédéré se réduit à un seul client. Pour 3, 5 et 10 robots, FedAvg Q-learning dépasse significativement l'apprentissage local indépendant sur le succès et le SPL (p corrigé ≤ 0,00391). Les gains de succès sont respectivement de 8,57, 5,90 et 6,56 points de pourcentage.

Face au Q-learning centralisé, FedAvg est meilleur à trois robots pour le succès (+2,55 points, p corrigé = 0,01172) et le SPL. À cinq robots, la différence de succès (+0,83 point) n'est pas significative (p corrigé = 0,22266), mais le SPL de FedAvg est supérieur (p corrigé = 0,01172). À dix robots, le centralisé possède un succès légèrement supérieur (+0,91 point, p corrigé = 0,01953), tandis que FedAvg conserve un SPL légèrement supérieur (p corrigé = 0,01953).

**Conclusion à retenir :** l'agrégation fédérée apporte un bénéfice net par rapport à l'absence de partage et obtient des performances globalement proches de l'apprentissage centralisé, sans nécessiter la centralisation des trajectoires locales.

## Comparaison des algorithmes dans l'architecture fédérée

| Robots | FedAvg Q-learning | FedAvg DQN | FedAvg PPO |
|---:|---:|---:|---:|
| 1 | **86,40 ± 1,62 %** | 78,65 ± 3,31 % | 10,25 ± 3,46 % |
| 3 | **73,48 ± 1,25 %** | 53,80 ± 2,18 % | 23,37 ± 4,96 % |
| 5 | **55,85 ± 1,10 %** | 38,67 ± 2,11 % | 20,29 ± 3,38 % |
| 10 | **41,02 ± 0,58 %** | 28,82 ± 0,56 % | 10,48 ± 1,29 % |

FedAvg Q-learning dépasse FedAvg DQN et FedAvg PPO pour toutes les tailles de flotte, sur le succès et le SPL, après correction de Holm. Cette observation est valable dans le protocole étudié : graphe discret de taille modérée, actions discrètes, budget d'entraînement identique et hyperparamètres retenus. Elle ne constitue pas une preuve de supériorité universelle du Q-learning.

## Scalabilité

Le succès de FedAvg Q-learning diminue de 86,40 % avec un robot à 41,02 % avec dix robots. Cette dégradation est associée à l'augmentation des conflits de mouvement, des collisions, de la congestion et des situations de blocage. Le projet évalue donc bien la scalabilité, mais les résultats ne permettent pas d'affirmer que la méthode élimine les difficultés de passage à l'échelle.

## Robustesse dynamique

| Robots | Statique, FedAvg Q | Fermeture dynamique | Baisse |
|---:|---:|---:|---:|
| 1 | 86,40 % | 72,05 % | 14,35 points |
| 3 | 73,48 % | 70,27 % | 3,22 points |
| 5 | 55,85 % | 54,44 % | 1,41 point |
| 10 | 41,02 % | 40,47 % | 0,55 point |

La diminution est significative pour chaque taille de flotte. Toutefois, le protocole ferme une arête partagée par scénario : l'effet relatif est mécaniquement plus dilué quand plusieurs robots sont évalués simultanément. Il ne faut donc pas conclure que la robustesse augmente avec le nombre de robots.

## Généralisation inter-bâtiments

Le transfert de *Office Building* vers *Clinic Architectural* reste faible. Le meilleur résultat observé est le DQN entraîné depuis zéro pendant 100 épisodes, avec environ 3,5 % de succès, contre environ 1,45 % en zero-shot et 1,85 % après 100 épisodes de fine-tuning. PPO reste proche de 1,3 %. Ces valeurs montrent que les représentations actuelles dépendent fortement du bâtiment source. Ce résultat doit être présenté comme une limite et comme une motivation pour une future représentation indépendante de la taille et de l'identité des nœuds.

## Formulations à éviter

- Ne pas écrire que FedAvg est toujours supérieur au centralisé.
- Ne pas écrire que Q-learning est universellement meilleur que DQN ou PPO.
- Ne pas écrire que la méthode généralise actuellement vers un nouveau bâtiment.
- Ne pas écrire que l'effet plus faible de la perturbation à dix robots prouve une meilleure robustesse.

## Fichiers de référence

- `table_A_architecture.csv` : comparaison local, centralisé et fédéré.
- `table_B_federated_algorithms.csv` : comparaison Q-learning, DQN et PPO dans FedAvg.
- `table_C_dynamic_robustness.csv` : performances dynamiques.
- `primary_paired_tests_across_seeds.csv` : tests principaux corrigés.
- `static_vs_dynamic_primary_tests.csv` : tests de robustesse corrigés.
- `figures/` : figures PNG et PDF prêtes à insérer.
