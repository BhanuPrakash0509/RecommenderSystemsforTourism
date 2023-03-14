from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy.spatial.distance import correlation
import pandas as pd
from numpy import linalg as la

popular_df = pickle.load(open('popularPlaces.pkl', 'rb'))

popular_df_city = pickle.load(open('popularPlacesOnCity.pkl', 'rb'))

df = pickle.load(open('df.pkl', 'rb'))

indices = pickle.load(open('indices.pkl', 'rb'))

cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

data = pickle.load(open('mergeData.pkl', 'rb'))

ratings_mat = pickle.load(open('ratings_mathybrid.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    data=''
    return render_template("index.html",
                           place_name=list(popular_df['Name'].values),
                           city_name=list(popular_df['City'].values),
                           desc=list(popular_df['Description'].values),
                           image=list(popular_df['Image'].values),
                           category=list(popular_df['Category'].values),
                           rating = list(popular_df['wr'].values),
                           data = data
                           )


@app.route('/popularitybasedoncity', methods=['post'])
def getpopularitybasedoncity():
    place_name = list(popular_df['Name'].values)
    city_name = list(popular_df['City'].values)
    desc = list(popular_df['Description'].values)
    image = list(popular_df['Image'].values)
    category = list(popular_df['Category'].values),
    rating = list(popular_df['wr'].values)

    cityName = request.form.get('city_input')
    if (str(cityName) != ""):
        data = popular_df_city[popular_df_city['City'].str.contains(cityName) |
                               popular_df_city['Category'].str.contains(cityName)]

        review_counts = data[data['Review_Count'].notnull()]['Review_Count'].astype('int')

        avg_rating = data[data['Avg_rating'].notnull()]['Avg_rating'].astype('int')

        mean_avg_rating = avg_rating.mean()
        min_review_count = review_counts.quantile(0.85)

        top_list = data[(data['Review_Count'] >= min_review_count) & (data['Review_Count'].notnull())
                        & (data['Avg_rating'] >= mean_avg_rating) & (data['Avg_rating'].notnull())][
            ['Name', 'Review_Count', 'Avg_rating', 'City', 'Description', 'Category', 'Image']]
        top_list['Review_Count'] = top_list['Review_Count'].astype('int')
        top_list['Avg_rating'] = top_list['Avg_rating'].astype('int')
        top_list['wr'] = top_list.apply(
            lambda x: (x['Review_Count'] / (x['Review_Count'] + min_review_count) * x['Avg_rating']) + (
                        min_review_count / (min_review_count + x['Review_Count']) * mean_avg_rating), axis=1)

        top_list = top_list.sort_values('wr', ascending=False).head(15)


        place_name = list(top_list['Name'].values)
        city_name = list(top_list['City'].values)
        desc = list(top_list['Description'].values)
        image = list(top_list['Image'].values)
        category = list(top_list['Category'].values)
        rating = list(top_list['wr'].values)
    return render_template("index.html",
                           place_name=place_name,
                           city_name=city_name,
                           desc=desc,
                           image=image,
                           category=category,
                           rating=rating
                           )

@app.route('/contentbased')
def recommend_ui():
    return render_template("contentbased.html")


@app.route('/getcontentbasedrecomm', methods=['post'])
def getrec():
    input = request.form.get('user_input')

    idx = indices[input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    place_indices = [i[0] for i in sim_scores]

    data = []
    for i in place_indices:
        item = []
        temp_df = df[df['Name'] == df['Name'].iloc[i]]
        item.extend(list(temp_df.drop_duplicates('Name')['Name'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['City'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Description'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Image'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Avg_rating'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Category'].values))
        data.append(item)

    return render_template("contentbased.html", data= data)


@app.route('/collobarativebased')
def collobarative_recommend():
    return render_template("collobarativebased.html")

userItemRatingMatrix=pd.pivot_table(data, values='Rating',index=['UserId'], columns=['ItemId'])

def getsimilarityOfUsers(user1, user2):
    user1 = np.array(user1) - np.nanmean(user1)
    user2 = np.array(user2) - np.nanmean(user2)
    commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]

    if len(commonItemIds) == 0:
        return 0
    else:
        user1 = np.array([user1[i] for i in commonItemIds])
        user2 = np.array([user2[i] for i in commonItemIds])
        return correlation(user1, user2)


def nearestNeighbourRatings(activeUser, K):
    similarityMatrix = pd.DataFrame(index=userItemRatingMatrix.index, columns=['Similarity'])

    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i] = getsimilarityOfUsers(userItemRatingMatrix.loc[int(activeUser)],
                                                       userItemRatingMatrix.loc[i])

    similarityMatrix = pd.DataFrame.sort_values(similarityMatrix, ['Similarity'], ascending=[0])

    nearestNeighbours = similarityMatrix[:K]

    neighbourItemRatings = userItemRatingMatrix.loc[nearestNeighbours.index]

    predictItemRating = pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])

    for i in userItemRatingMatrix.columns:
        predictedRating = np.nanmean(userItemRatingMatrix.loc[int(activeUser)])

        for j in neighbourItemRatings.index:
            if userItemRatingMatrix.loc[j, i] > 0:
                predictedRating += (userItemRatingMatrix.loc[j, i] - np.nanmean(userItemRatingMatrix.loc[j])) * \
                                   nearestNeighbours.loc[j, 'Similarity']
            predictItemRating.loc[i, 'Rating'] = predictedRating
    return predictItemRating


def topNRecommendations(activeUser, N):
    predictItemRating = nearestNeighbourRatings(int(activeUser), N)

    placeAlreadyWatched = list(
        userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser] > 0].index)

    predictItemRating = predictItemRating.drop(placeAlreadyWatched)

    topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:N]

    topRecommendationTitles = (df.loc[df.ItemId.isin(topRecommendations.index)])

    data = []
    for name in topRecommendationTitles.Name:
        item = []
        temp_df = topRecommendationTitles[topRecommendationTitles['Name'] == name]
        item.extend(list(temp_df.drop_duplicates('Name')['Name'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['City'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Description'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Image'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Avg_rating'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Category'].values))
        data.append(item)
    return data


@app.route('/getuserbasedrecomm', methods=['post'])
def getuserbasedrec():
    userId = request.form.get('user_id')

    data = topNRecommendations(int(userId), 10)

    return render_template("collobarativebased.html", data=data)


@app.route('/hybridmodel')
def hybrid_recommend():
    return render_template("hybridmodel.html")

def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def get_k(sigma, percentage):
    sigma_sqr=sigma**2
    sum_sigma_sqr=sum(sigma_sqr)
    k_sum_sigma=0
    k=0
    for i in sigma:
        k_sum_sigma+=i**2
        k+=1
        if k_sum_sigma>=sum_sigma_sqr*percentage:
            return k

def svdEst(testdata, user, simMeas, item, percentage):
    n=np.shape(testdata)[1]
    sim_total=0.0;
    rat_sim_total=0.0
    u,sigma,vt=la.svd(testdata)
    k=get_k(sigma,percentage)
    #Construct the diagonal matrix
    sigma_k=np.diag(sigma[:k])
    #Convert the original data to k-dimensional space (lower dimension) according to the value of k. formed_items represents the value of item in k-dimensional space after conversion.
    formed_items=np.around(np.dot(np.dot(u[:,:k], sigma_k),vt[:k, :]),decimals=3)
    for j in range(n):
        user_rating=testdata[user,j]
        if user_rating==0 or j==item:continue
        # the similarity between item and item j
        similarity=simMeas(formed_items[item,:].T,formed_items[j,:].T)
        sim_total+=similarity
        # product of similarity and the rating of user to item j, then sum
        rat_sim_total+=similarity*user_rating
    if sim_total==0:
        return 0
    else:
        return np.round(rat_sim_total/sim_total, decimals=3)

#Predicted ratings
def recommend(testdata, user, sim_meas, est_method, percentage=0.9):
    unrated_items=np.nonzero(testdata[user,:]==0)[0].tolist()
    if len(unrated_items)==0:
        return print('everything is rated')
    item_scores=[]
    for item in unrated_items:
        estimated_score=est_method(testdata,user,sim_meas,item,percentage)
        item_scores.append((item,estimated_score))
    item_scores=sorted(item_scores,key=lambda x:x[1],reverse=True)

    data=[]
    for i in item_scores:
        item=[]
        temp_df  = df[df['ItemId'].isin(i)]
        print(temp_df)
        item.extend(list(temp_df.drop_duplicates('Name')['Name'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['City'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Category'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Description'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Image'].values))
        item.extend(list(temp_df.drop_duplicates('Name')['Avg_rating'].values))
        data.append(item)
    return data

@app.route('/gethybridmodelrecomm', methods=['post'])
def gethybridrec():
    userId = request.form.get('user_id')

    data = topNRecommendations(int(userId), 10)

    return render_template("hybridmodel.html", data=data)

if __name__ == '__main__':
    app.run(debug=True)