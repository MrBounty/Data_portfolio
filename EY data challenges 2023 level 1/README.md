# Introduction
The EY Open Science Data Challenge Level 1 aims to create open-source solutions to address UN Sustainable Development Goal 2: Zero Hunger. Participants will use radar and optical satellite data to build a model that identifies rice crops in Vietnam. Vietnam is a leading rice producer and is highly vulnerable to climate change.   


The challenge is scalable and has the potential to transform global rice production. The challenge requires math and coding skills and completing it will improve skills in Python for data science, machine learning, and managing large volumes of data. Participants will predict the presence of rice crops in a specific province of Vietnam using satellite data, with opportunities for improvement including choice of data and dataset, data preprocessing, model design and hyperparameter tuning. The output will be a rice crop classification model that can prioritize actions to ensure food security.

# My solution
I created my solution to identify rice crops in Vietnam using deep learning technologies with Keras. Using data solely from the satellite Sentinel 1, my model achieved an accuracy of 100%. It took me around 30 hours to obtain a model with 100% accuracy, but I also spend around 20 hours making optimization and visualization.  

In this section I will explain the workflow I used to make my model. Starting with what data I use and how I imported it. Then explain the transformation and preprocessing that I applied to it. To continue with a presentation of my Deep Learning model and how it has been train. To finish by quickly explain how I used my model to make prediction and showing a visualization of what my model is capable of.

## Import the data
Choosing the right data in a machine learning problem is arguably the most important step. It can greatly impact the accuracy and effectiveness of the resulting model. In this particular case, the challenge involved using data from multiple satellites to make predictions. The different data sources included temperature readings, vegetation indices, and others. However, after trying out various combinations, the decision was made to focus solely on RVI data (a vegetation indices). This was due to the fact that cloud interference was minimized, and the resulting curve was consistent. Additionally, the RVI data provided enough information to make accurate predictions. By carefully selecting the most appropriate data, the resulting model was able to achieve better performance and generate more meaningful insights.

Here the workflow of how I import the data that my model will use:
*	Load the csv file with 2 columns: coordinate (latitude and longitude) and rice crop presence
*	Define a square box of a chosen size around each coordinate
*	Use the box to extract data from Sentinel 1 satellite for all available dates in 2022
*	Obtain an image of radio wave values and calculate the vegetation index (RVI) to represent the amount of vegetation.
*	Compute the mean value of the RVI image
*	For each coordinate, obtain an array of RVI values and an array of dates when these values were taken.

Here an image of the window use to calculate the mean RVI (the one with less resolution). And another one to give an idea of the size of it, the red box is the window use:
![alt text](https://github.com/MrBounty/Data_portfolio/blob/main/EY%20data%20challenges%202023%20level%201/image/Picture1.png)

Here an plot of the mean RVI value over time for coordinate with and without rice:   
As we can see on those graph, coordinate with rice follow a periodic curve because of the life cycle of the rice as presented in figure xx.  That is what my model will try to use to predict if there is rice at a certain coordinate. Also, during my experimentation, I tried to use a polynomial interpolation because of the shape of the curve but I did not obtain a satisfactory result.
![alt text](https://github.com/MrBounty/Data_portfolio/blob/main/EY%20data%20challenges%202023%20level%201/image/Picture3.png)
![alt text](https://github.com/MrBounty/Data_portfolio/blob/main/EY%20data%20challenges%202023%20level%201/image/Picture4.png)
