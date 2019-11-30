# Comment profiter au mieux du contenu de ce projet

Si la seule lecture de l'étude vous convient, c'est par [la](./../README.md).

Pour ceux qui veulent avoir accès à tensorboard pour visualiser tous les graphiques au mieux, suivez les instructions suivantes.


- Créez un nouveau répertoire.
- <code>git clone https://github.com/TimGuez/Predictive_keyboard.git</code>
- Installer tous les packages nécessaires avec <code> pip install -r requirements.txt</code>
- Lancer tensorboard avec <code> tensorboard --logdir=model_lstm_hp_tuning_logs</code>

Vous pouvez également lancer les scripts python de votre choix : 
- *live_test_word* permet de rentrer une phrase et de voir les prédiction associées
- *kspc* calcule la valeur Key Stroke Per Character sur le jeu de données
- *train_processor* entraine un nouveau preprocessor
- *hp_tuning* lance une optimisations d'hyperparamètre sur le modèle LSTM bidirectionnel

