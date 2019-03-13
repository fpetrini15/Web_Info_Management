from scipy import spatial, stats
from sklearn import preprocessing
import math
from collections import OrderedDict 
import random

#dictionary to count to total occurrences each rating
totals = {
	1: 0,
	2: 0,
	3: 0,
	4: 0,
	5: 0
}

train_ratings = {
	1:0,
	2:0,
	3:0,
	4:0,
	5:0
}

methods = {
	"cosine-similarity": 0,
	"pearson-correlation": 1,
	"item-based": 2,
	"semi-random": 3
}

#value to set the no. of strikes before tossing out vector
num = 8

search_index = [287, 293, 329, 357, 876]
predictions = []
movie_popularity = OrderedDict()

def generateMoviePopularity():
	for i in range(0, 1000):
		movie_popularity[i] = [0,0]


#this function goes through the training data and creates a large dictionary that stores
#data in the following format: users: {user1:[r1, r2, ],user2[]} where the userID is the
#key and the corresponding list is an array of ratings (indexed by movie)
def gatherData(filename):
	i = -1
	users = {}		
	for line in filename:
		i += 1
		users[i] = []
		line = [int(x) for x in line.split()]
		users[i] = line
	return users

def gatherPopularity(train_data):
	#import pdb; pdb.set_trace()
	for user in train_data.keys():
		li = train_data[user]
		for movie, rating in enumerate(li):
			if rating != 0:
				movie_popularity[movie][0] += rating
				movie_popularity[movie][1] += 1
				train_ratings[rating] += 1

def generateIDF(IDF_train):
	factors = []
	for each in movie_popularity:
		if movie_popularity[each] == 0:
			idf = 0
		else:
			if movie_popularity[each][1] == 0:
				idf = 0
			else:
				idf = math.log(1000/(movie_popularity[each][1]), 2)
		'''
		for i in range(0, len(IDF_train)):
			IDF_train[i][each] = IDF_train[i][each]*idf
		'''
		factors.append(idf)
	return factors

#this function opens each test file and creates a dictionary of new users. It scans each
#line. If line[0] (userID) is not in the dictionary, it creates a new list with the userID
#as the key. It then fills the corresponding list with tuples (movie, rating). The result
#is new_users = {201: [(237,4), (306,5)...]}
def createUser():
	filename = open("test5.txt")
	new_users = {}
	for line in filename:
		line = [int(x) for x in line.split()]			
		if line[0] not in new_users:					
			new_users[line[0]] = []						
			#import pdb; pdb.set_trace()
		new_users[line[0]].append((line[1]-1, line[2]))
	return new_users

#this function goes through all the newly created users. It separates the tuples from new_users
#for each userID into those where the rating was provided and those that need to be predicted. For
#the values that need to be predicted, the ratedEntries are passed with the training data in order
#to assign a rating. When similars is returned, an average is generated and a prediction is assigned.
def findSimilarity(train_data, new_users, idf_factors):
	similarityList = {}
	result = []
	i = 0
	for key in new_users.keys():		 
		user = new_users[key]			
		ratedEntries, ratedEntryIndicies = [], []
		predict = []
		
		for rating in user:				
			if(rating[1] != 0):			
				ratedEntries.append(rating)	
				ratedEntryIndicies.append(rating[0])
			else: 
				predict.append(rating[0])	
				predictions.append([str(key), str(rating[0]+1), str(0)])

		#method = "cosine-similarity"
		#method = "pearson-correlation"
		#method = "item-based"
		method = "semi-random"
		#import pdb; pdb.set_trace()
		if methods[method] != 3:
			similars, new_user_rating = extractSimilars(train_data, ratedEntries, idf_factors, method, ratedEntryIndicies)
			result = makePredictions(similars[::-1], predict, train_data, method, new_user_rating)
		else:
			result = makePredictions([], predict, train_data, method, [])
		for each in result:
			predictions[i][2] = str(each)
			i+=1
	
#this function takes all the preprocessed data and applies methods of calculating similarities
#in order to generate reasonable predictions for unknown fields. For each key in the training
#data, it creates two arrays: one to store ratings from the newly created users and one to 
#store ratings from the users from the training data. It then iterates through the tuples of rated
#entries using movies as the index and finds the corresponding lists for each userID in the train_data
#A similarity function is applied and the resulting list is sorted for later use.
def extractSimilars(train_data, userRatings, idf_factors, method, ratedEntries):
	similarityList = []
	for key in train_data.keys():		
		train_rating = []
		new_user_rating = []
		for tup in userRatings:
			#train_rating.append(train_data[key][tup[0]]*idf_factors[tup[0]])
			train_rating.append(train_data[key][tup[0]])
			new_user_rating.append(tup[1])
		if methods[method] == 0:
			result = calcCosineSimilarity(train_rating, new_user_rating)
			if result == 0:
				continue;
			temp = train_rating.copy()
			similarityList.append((result, key, temp))
		elif methods[method] == 1:
			result, simUserAvg, uToPredictAvg = pearsonCorrelationWeights(train_rating, new_user_rating)
			if result == 0:
				continue;
			temp = train_rating.copy()
			similarityList.append((result, key, simUserAvg, uToPredictAvg, temp))
		elif methods[method] == 2:
			result = itemBasedCosineSimilarity(train_rating, new_user_rating, ratedEntries)
			if result == 0:
				continue;
			temp = train_rating.copy()
			similarityList.append((result, key, temp))
	similarityList.sort()
	
	return similarityList, new_user_rating

def makePredictions(similars, predict_movie_indicies, train_data, method, new_user_rating):
	res = []
	if methods[method] == 3:
		for index in predict_movie_indicies:
			val = generateSemiRandom()
			totals[val] += 1
			res.append(val)
		return res
	for index in predict_movie_indicies:
		sim_values, simUsers, sim_avg = [], [], []
		avg, count, i, prev, threshold = 0, 0, 0, 0, 0.8
		while count != 15:
			#if you run out of data
			if(i >= len(similars)):
				break
			#if you cross the threshold
			
			if(similars[i][0] < threshold):
				#import pdb; pdb.set_trace()
				break
			#get the rating assigned to this movie from the ith similar user
			val = train_data[similars[i][1]][index]
			#if the ith user hasn't seen the movie, continue to the next user
			if(val == 0):
				i += 1
				continue
			else:
				#cosine similarity
				if methods[method] == 0:
					#otherwise add it to the list of most similar users
					simUsers.append(train_data[similars[i][1]][index])
					sim_values.append(similars[i][0])
				#pearson correlation
				elif methods[method] == 1:
					#rating
					simUsers.append(train_data[similars[i][1]][index])
					#similarity weight
					sim_values.append(similars[i][0])
					#averge
					sim_avg.append(similars[i][2])
				elif methods[method] == 2:
					simUsers.append(train_data[similars[i][1]][index])
					sim_values.append(similars[i][0])
				count += 1
				i += 1
		if(simUsers == []):
			if movie_popularity[index][1] == 0:
				totals[3] += 1
				res.append(3)
			else:
				val = round(movie_popularity[index][0]/movie_popularity[index][1])
				totals[val] += 1
				res.append(val)
		else:
			if methods[method] == 0 or methods[method] == 2:
				val = round(predictCosineSimilarityBased(simUsers, sim_values))
			elif methods[method] == 1:
				val = round(pearsonCorrelationPrediction(simUsers, new_user_rating, sim_values, sim_avg, 0))
			if val < 1:
				val = 1
			elif val > 5:
				val = 5
			totals[val] += 1
			res.append(val)
	return res


def calcCosineSimilarity(simUser, userToPredict):
	numerator, denom1, denom2, count = 0, 0, 0, 0 
	#generate the numerator
	#import pdb; pdb.set_trace()
	for each in simUser:
		if each == 0:
			count += 1
		if count == num:
			return 0 
	for i in range(0, len(simUser)):
		if(simUser[0] == 0):
			continue
		numerator += simUser[i] * userToPredict[i]
	#generate the denominator for the similar user
	for i in range(0, len(simUser)):
		if simUser[i] == 0:
			continue
		denom1 += simUser[i]**2
	denom1 = math.sqrt(denom1)
	#generate the denominator for the new user
	for i in range(0, len(userToPredict)):
		if simUser[i] == 0:
			continue
		denom2 += userToPredict[i]**2
	denom2 = math.sqrt(denom2)
	#sum the denominators and return the value
	denominator = denom1 * denom2

	if(denominator != 0):
		return numerator / denominator
	else:
		return 0

#simUsers-> a list of similar users
#similarity->a list of similarity values for corresponding simUser & userToPredict
def predictCosineSimilarityBased(simUsers, similarity):
	numerator, denominator = 0, 0
	for index, rating in enumerate(simUsers):
		numerator += similarity[index]*rating
		denominator += similarity[index]
	if(denominator != 0):
		return numerator / denominator
	else:
		return 0

def pearsonCorrelationWeights(simUser, userToPredict):
	#import pdb; pdb.set_trace()
	simUserAvg, uToPredictAvg, count = 0, 0, 0
	for each in simUser:
		if each == 0:
			count += 1
		if count == num:
			return 0, 0, 0
	#calculate average rating for the similar user
	#import pdb; pdb.set_trace()
	simUserAvg = generateAverage(simUser)
	#calculate average rating for the new user
	uToPredictAvg = generateAverage(userToPredict)
	if simUserAvg == 0:
		return 0, 0, 0
	numerator, denom1, denom2 = 0, 0, 0
	#generate the numerator

	for i in range(0, len(simUser)):
		if simUser[i] == 0:
			continue
		numerator += (userToPredict[i] - uToPredictAvg)*(simUser[i] - simUserAvg)
	#generate the denominator for the new user portion
	for i in range(0, len(userToPredict)):
		if simUser[i] == 0:
			continue
		denom1 += (userToPredict[i] - uToPredictAvg)**2
	denom1 = math.sqrt(denom1)
	#generate the denominator for the similar user portion
	for i in range(0, len(simUser)):
		if simUser[i] == 0:
			continue
		denom2 += (simUser[i] - simUserAvg)**2
	denom2 = math.sqrt(denom2)
	#sum the denominators and return the result
	denominator = denom1 * denom2
	
	if(denominator != 0):
		return (numerator / denominator), simUserAvg, uToPredictAvg
	else:
		return 0, 0, 0

#simUsers -> a list of similar users with their corresponding ratings for a particular movie
#userToPreidct -> the new user that needs a rating for a movie assigned
#weigts -> a list of weights associated with each simUser
#averages -> list of corresponding averages

def pearsonCorrelationPrediction(simUserRatings, userToPredict, weights, averages, caseMod):
	uToPredictAvg, simUserAvg, count = 0, 0, 0
	#generate the average for the new user whose value needs to be predicted
	uToPredictAvg = generateAverage(userToPredict)
	#if we want to use case modification
	if(caseMod):
		print("here")
		weights = caseModification(weights)
	
	#generate the numerator and denominator
	numerator, denominator = 0, 0
	for i in range(0, len(simUserRatings)):
		#numerator requires corresponding weight and a specific user's rating subtracted
		#by that user's average rating
		numerator += weights[i]*(simUserRatings[i] - averages[i])
		#denominator requires the corresponding weights as a positive number
		denominator += abs(weights[i])

	return uToPredictAvg + (numerator/denominator)

def caseModification(weights):
	new_weights = []
	for weight in weights:
		new_weights.append(weight * abs(weight)**2.5)
	return new_weights

def itemBasedCosineSimilarity(simUser, userToPredict, predict_movie_indicies):
	numerator, denom1, denom2 = 0, 0, 0
	globalAvg = []
	#import pdb; pdb.set_trace()
	for index in predict_movie_indicies:
		if movie_popularity[index][1] == 0:
			globalAvg.append(3)
		else:
			globalAvg.append(movie_popularity[index][0]/movie_popularity[index][1])
	#generate the numerator
	for i in range(0, len(simUser)):
		if simUser[i] == 0:
			continue
		numerator += (simUser[i]-globalAvg[i]) * (userToPredict[i]-globalAvg[i])
	#generate the denominator for the similar user
	for i in range(0, len(simUser)):
		if simUser[i] == 0:
			continue
		denom1 += (simUser[i]-globalAvg[i])**2
	denom1 = math.sqrt(denom1)
	#generate the denominator for the new user
	for i in range(0, len(userToPredict)):
		if simUser[i] == 0:
			continue
		denom2 += (userToPredict[i]-globalAvg[i])**2
	denom2 = math.sqrt(denom2)
	#sum the denominators and return the value
	denominator = denom1 * denom2

	if(denominator != 0):
		return numerator / denominator
	else:
		return 0

def generateSemiRandom():
	import random
	probability_list = []
	factor = 1
	for each in train_ratings.keys():
		if each == 4:
			factor = 10
		elif each == 3:
			factor = 4
		elif each == 5:
			factor = 3
		else:
			factor = 2
		#import pdb; pdb.set_trace()
		probability_list = probability_list + [each]*(train_ratings[each]*factor)
	return random.choice(probability_list)

def generateAverage(user):
	avg, count = 0, 0
	for rating in user:
		if rating == 0:
			continue
		else:
			avg += rating
			count += 1
	if(count != 0):
		return avg/count
	else:
		return 0

def writeResult():
	file = open("test5_result.txt","w+")
	predictions.sort()
	for each in predictions:
		#import pdb; pdb.set_trace()
		each = ' '.join(each)
		file.write(each)
		file.write("\n")


filename = open("train.txt")
generateMoviePopularity()
train = gatherData(filename)
gatherPopularity(train.copy())
idf_factors = generateIDF(train.copy())
new_users = createUser()
findSimilarity(train, new_users, idf_factors)
writeResult()
#print(movie_popularity)
print(train_ratings)
print(totals)



