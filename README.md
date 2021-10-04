## CMPE255 Team Project

#Player Action Prediction for ROCK – PAPER – SCISSORS 

#### Team Members:

* Akshay Jain 
* Nidhi Tholar [nidhi-tholar](https://github.com/nidhi-tholar)
* Sashank

### DESCRIPTION OF THE PROBLEM:
The project aims to provide a modern approach using Data Mining techniques for a classical power balanced game. As described generally, the game of Rock-Paper-Scissors is played between two with total of 9 possibilities and winning chances of 1 in 3. However, when with humans involved in the game, the chance of randomness diminishes as humans tend to repeat any set of winning moves. With the use of pattern identification and prediction techniques, the odds of winning for a given set of games or plays drastically increases. The project also aims in incorporating the lecture topics taught in realizing and meeting the project outcomes. The end goal of the project is to successfully predict the player’s moves to win the game.

### Considered Data:
The project topic considered on the game of Rock Paper Scissors, and the supporting dataset which is considered for the project is provided in Kaggle by user Nikos Koumbakis. The dataset is mentioned to be created as a play between 2 agents mimicking the players for the given set of actions (Rock, Paper, Scissors). The dataset consists of games (a season of game) with a set of rounds (episodes) per game between 2 players with the outcome of the round being win/lose/draw. This outcome of the round is noted as reward for the agent. This dataset consists of player’s moves that is slightly biased for one player to mimic a pattern form of a human. The dataset could be found in the link Rock Paper Scissors Agents Battles | Kaggle 

### Methodology and Approach:
Preliminary process includes determining the power balance in the action moves in the dataset and determining if the dataset is clean. Further, the usage of Identifying Frequent Itemsets, Naïve Bayes Classification to determine pattern of win-lose-draw in the played games dataset. From the determined move, A simple decision tree could be incorporated to make the computer’s move to result the outcome of the round. Though the considered methodology on determining and mining might change in actual implementation, the initial phase of exploration suggests the considered methods as a possibility.

### Success Determination:
The Game’s perspective suggests considering only true wins as success and true lose as failure with minimizing the outcome of draws. From the initially suggested techniques to determine and predict the player’s next move, a simple ratio could determine the success rate of the game. Though the dataset is used to have intuition on the previously played moves, splitting the same dataset to have a testing set could provide scope to evaluate our model’s accuracy.

### Other Notable Links for Inspiration:
	Online Rock-Paper-Scissors AI based Game portal : [Rock, Paper, Scissors | Afiniti] (https://www.afiniti.com/corporate/rock-paper-scissors)
