# 🔍 Power Math Crawler & BM25 Search Engine (Moteur de Recherche Mathématique)

Ce projet implémente un système complet de récupération d'information spécialisé dans le contenu mathématique. Il combine un crawler récursif performant basé sur `crawl4ai` pour la collecte de données sur des sites comme Wikipédia, avec un moteur de recherche sophistiqué utilisant l'algorithme de classement **BM25 (Best Match 25)** et la **pondération des champs** pour des résultats d'une pertinence supérieure au TF-IDF standard.

## ✨ Fonctionnalités Clés

* **Crawling Récursif (BFS):** Utilise `crawl4ai` pour explorer et indexer automatiquement les pages liées à partir d'une liste de graines (seed URLs), se concentrant sur les domaines pertinents (ex: `wikipedia.org`).
* **Extraction Structurée:** Emploie une stratégie d'extraction CSS ciblée pour séparer les composants clés du contenu mathématique :
    * Titre (`title`, `h1`)
    * Contenu textuel (`p`, `li`)
    * **Formules et Équations** (via des sélecteurs comme `.katex`, `.mwe-math-element`).
* **Classement BM25 Pondéré :** Le cœur du moteur de recherche. Il utilise l'algorithme BM25, en attribuant un poids supérieur aux correspondances trouvées dans les champs `title` et `formulas` pour maximiser la pertinence des théorèmes et définitions. 
* **Indexation Persistante:** Utilise `SQLite3` pour stocker durablement les documents, les métadonnées et les liens, permettant une recherche rapide sans recrawl à chaque exécution.

## 🛠️ Prérequis

Pour exécuter ce projet, vous devez disposer de Python 3.x et installer les dépendances suivantes :

```bash
pip install crawl4ai
