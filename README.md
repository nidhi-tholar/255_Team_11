# 255_Team_11
This Repository has been created for the Final Project of CMPE-255.


Team Members

Akshay Jain

Nidhi Tholar [nidhi-tholar]

Sashank

Project title: ROCK – PAPER – SCISSORS

Description of the problem:
This project aims to provide a modern approach for a classical power balanced contest. As known, the game of Rock-Paper-Scissors itself is played between two with total of 9 possibilities and chance of winning is 1 in 3. However, with humans involved, the chance of randomness diminishes as humans tend to repeat any set of winning moves and use of pattern identification and prediction techniques greatly increases the odds of winning for a given set of games or plays. This Project also aims in incorporating the lecture topics taught in realizing the outcomes

Data:
Since the project is on the topic of Rock Paper Scissors, the data set which we are considering to use in the project is provided in Kaggle by user Nikos Koumbakis. The DataSet consists of games (a season of game) with a set of rounds (episodes) per game between 2 players with the outcome of the round. This Dataset consists of player’s moves that is slightly biased for one player to mimic a pattern form of a human. The Data could be found in the link Rock Paper Scissors Agents Battles | Kaggle

Methodology:
	Preliminary process includes determining the power balance in the action moves in the dataset and determining if the dataset is clean. Further, the usage of Identifying Frequent Itemsets, Naïve Bayes Classification to determine pattern of win-lose-draw in the played games dataset. From the determined move, A simple decision tree could be incorporated to make the computer’s move to result the outcome of the round. Though the considered methodology on determining and mining might change in actual implementation, the initial phase of exploration suggests the considered methods as a possibility.
  
 Determination of Success:
	The Game’s perspective suggests considering only true wins as success and true lose as failure with minimizing the outcome of draws. From the initially suggested techniques to determine and predict the player’s next move, a simple confusion matrix provides a visualization on the round’s outcomes.

