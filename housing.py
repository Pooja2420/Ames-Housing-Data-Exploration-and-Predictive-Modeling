#!/usr/bin/env python
# coding: utf-8

# # Project Description.

# For this project we will be exploring publicly available data from https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data.
# 
# This project involves analyzing the Ames Housing dataset, which contains detailed information on residential homes in Ames, Iowa. The dataset is often used for predictive modeling and includes 82 columns with various attributes describing the properties, such as their physical characteristics, sale details, and neighborhood information. 
# 
# Here are what the columns represent:
# 
# MSSubClass: Identifies the type of dwelling involved in the sale.	
# 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
# 
# MSZoning: Identifies the general zoning classification of the sale.
# 		
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
# 	
# LotFrontage: Linear feet of street connected to property
# 
# LotArea: Lot size in square feet
# 
# Street: Type of road access to property
# 
#        Grvl	Gravel	
#        Pave	Paved
#        	
# Alley: Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
# 		
# LotShape: General shape of property
# 
#        Reg	Regular	
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular
#        
# LandContour: Flatness of the property
# 
#        Lvl	Near Flat/Level	
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
# 		
# Utilities: Type of utilities available
# 		
#        AllPub	All public Utilities (E,G,W,& S)	
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only	
# 	
# LotConfig: Lot configuration
# 
#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property
# 	
# LandSlope: Slope of property
# 		
#        Gtl	Gentle slope
#        Mod	Moderate Slope	
#        Sev	Severe Slope
# 	
# Neighborhood: Physical locations within Ames city limits
# 
#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	Northwest Ames
#        OldTown	Old Town
#        SWISU	South & West of Iowa State University
#        Sawyer	Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	Timberland
#        Veenker	Veenker
# 			
# Condition1: Proximity to various conditions
# 	
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 	
# Condition2: Proximity to various conditions (if more than one is present)
# 		
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 	
# BldgType: Type of dwelling
# 		
#        1Fam	Single-family Detached	
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit
# 	
# HouseStyle: Style of dwelling
# 	
#        1Story	One story
#        1.5Fin	One and one-half story: 2nd level finished
#        1.5Unf	One and one-half story: 2nd level unfinished
#        2Story	Two story
#        2.5Fin	Two and one-half story: 2nd level finished
#        2.5Unf	Two and one-half story: 2nd level unfinished
#        SFoyer	Split Foyer
#        SLvl	Split Level
# 	
# OverallQual: Rates the overall material and finish of the house
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 	
# OverallCond: Rates the overall condition of the house
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average	
#        5	Average
#        4	Below Average	
#        3	Fair
#        2	Poor
#        1	Very Poor
# 		
# YearBuilt: Original construction date
# 
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 
# RoofStyle: Type of roof
# 
#        Flat	Flat
#        Gable	Gable
#        Gambrel	Gabrel (Barn)
#        Hip	Hip
#        Mansard	Mansard
#        Shed	Shed
# 		
# RoofMatl: Roof material
# 
#        ClyTile	Clay or Tile
#        CompShg	Standard (Composite) Shingle
#        Membran	Membrane
#        Metal	Metal
#        Roll	Roll
#        Tar&Grv	Gravel & Tar
#        WdShake	Wood Shakes
#        WdShngl	Wood Shingles
# 		
# Exterior1st: Exterior covering on house
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast	
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# Exterior2nd: Exterior covering on house (if more than one material)
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# MasVnrType: Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
# 	
# MasVnrArea: Masonry veneer area in square feet
# 
# ExterQual: Evaluates the quality of the material on the exterior 
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# ExterCond: Evaluates the present condition of the material on the exterior
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# Foundation: Type of foundation
# 		
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete	
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood
# 		
# BsmtQual: Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# BsmtCond: Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# BsmtExposure: Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# BsmtFinType1: Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 		
# BsmtFinSF1: Type 1 finished square feet
# 
# BsmtFinType2: Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 
# BsmtFinSF2: Type 2 finished square feet
# 
# BsmtUnfSF: Unfinished square feet of basement area
# 
# TotalBsmtSF: Total square feet of basement area
# 
# Heating: Type of heating
# 		
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace	
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace
# 		
# HeatingQC: Heating quality and condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# CentralAir: Central air conditioning
# 
#        N	No
#        Y	Yes
# 		
# Electrical: Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
# 		
# 1stFlrSF: First Floor square feet
#  
# 2ndFlrSF: Second floor square feet
# 
# LowQualFinSF: Low quality finished square feet (all floors)
# 
# GrLivArea: Above grade (ground) living area square feet
# 
# BsmtFullBath: Basement full bathrooms
# 
# BsmtHalfBath: Basement half bathrooms
# 
# FullBath: Full bathrooms above grade
# 
# HalfBath: Half baths above grade
# 
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
# 
# Kitchen: Kitchens above grade
# 
# KitchenQual: Kitchen quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        	
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 
# Functional: Home functionality (Assume typical unless deductions are warranted)
# 
#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only
# 		
# Fireplaces: Number of fireplaces
# 
# FireplaceQu: Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
# 		
# GarageType: Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# 		
# GarageYrBlt: Year garage was built
# 		
# GarageFinish: Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage
# 		
# GarageCars: Size of garage in car capacity
# 
# GarageArea: Size of garage in square feet
# 
# GarageQual: Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# GarageCond: Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# PavedDrive: Paved driveway
# 
#        Y	Paved 
#        P	Partial Pavement
#        N	Dirt/Gravel
# 		
# WoodDeckSF: Wood deck area in square feet
# 
# OpenPorchSF: Open porch area in square feet
# 
# EnclosedPorch: Enclosed porch area in square feet
# 
# 3SsnPorch: Three season porch area in square feet
# 
# ScreenPorch: Screen porch area in square feet
# 
# PoolArea: Pool area in square feet
# 
# PoolQC: Pool quality
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
# 		
# Fence: Fence quality
# 		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
# 	
# MiscFeature: Miscellaneous feature not covered in other categories
# 		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
# 		
# MiscVal: $Value of miscellaneous feature
# 
# MoSold: Month Sold (MM)
# 
# YrSold: Year Sold (YYYY)
# 
# SaleType: Type of sale
# 		
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
# 		
# SaleCondition: Condition of sale
# 
#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
# 

# 
# ### Import Required libraries.

# In[37]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ### Get the data
# ### Using pandas to read in the datasets as a dataframe

# In[2]:


house_df = pd.read_csv("AmesHousing.csv")


# In[3]:


house_df


# In[4]:


house_df.describe(include = 'all').T


# In[5]:


house_df.info()


# ### Looking for mising values

# In[6]:


missing_values = house_df.isnull().sum()

# Filtering columns with missing values.
columns_with_missing_values = missing_values[missing_values > 0]

# Displaying columns with missing values.
print(columns_with_missing_values)


# Checking for percentage of null values and droping the column have more than 30% of missing values in it, since 25-30% of missing data is okay to proceed with.

# In[7]:


null_cols_list=['Lot Frontage', 'Alley', 'Mas Vnr Type', 'Mas Vnr Area', 'Bsmt Qual',
       'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
       'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
       'Electrical', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Fireplace Qu',
       'Garage Type', 'Garage Yr Blt', 'Garage Finish', 'Garage Cars',
       'Garage Area', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence',
       'Misc Feature']
for colName in null_cols_list:
    print('The percentage of null values in column:',colName,'-',round((house_df[colName].isnull().sum()/house_df.shape[0])*100,2))
    if(round((house_df[colName].isnull().sum()/house_df.shape[0])*100,2)>30):
        house_df.drop([colName],axis=1,inplace=True)
        print('Column ',colName,' dropped because percentage of null values was greater than 30%')


# In[8]:


house_df.shape


# # Handling Missing Data
# Since there is a large number of missing data, we are not removing the entire row instead filling the missing values with **mean** for numerical columns and **mode** for categorical columns.

# ### Numerical Columns

# In[9]:


for col in house_df.select_dtypes(include=[np.number]).columns:
    house_df[col].fillna(house_df[col].mean(), inplace=True)


# ### Plots for Numerical variables

# In[10]:


num_vars = ['MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 'SalePrice']
for var in num_vars:
    sns.histplot(house_df[var], kde=True)
    plt.show()


# ### Categorical Columns

# In[11]:


for col in house_df.select_dtypes(include=[object]).columns:
    house_df[col].fillna(house_df[col].mode()[0], inplace=True)


# ### Plots for Categorical Variables

# In[12]:


cat_vars = ['MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC', 'Central Air', 'Electrical', 'Kitchen Qual', 'Functional', 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive', 'Sale Type', 'Sale Condition']
for var in cat_vars:
    sns.countplot(y = var, data = house_df)
    plt.show()


# ### Dropping the PID column.

# In[13]:


df = house_df.drop(columns=['PID'])
df.info()


# # Skewness

# In[14]:


skewness = df.skew()
print("Skewness in DataFrame:")
print(skewness)


# # Visualization of Skewness

# In[15]:


plt.figure(figsize=(10, 8))
skewness.plot(kind='bar')
plt.title('Skewness of Ames Housing Data Set')
plt.ylabel('Skewness')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # Correlation

# In[16]:


# Correlation with SalePrice
correlation_matrix = df.corr()
saleprice_correlation = correlation_matrix['SalePrice'].sort_values(ascending=False)
saleprice_correlation


# In[17]:


# Plot correlation heatmap for the top 10 features
top_10_features = saleprice_correlation.index[1:11]  # Excluding SalePrice itself
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_10_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap with SalePrice')
plt.show()


# The heatmap is a useful tool for quickly identifying which features are most strongly associated with SalePrice, aiding in feature selection for predictive modeling.
# 
# **Strong Positive Correlation:** There’s a strong positive correlation between SalePrice and Overall Qual, indicating higher quality houses tend to sell for more.
# 
# **High Correlation with Garage:** Both Garage Cars and Garage Area have high positive correlations with SalePrice, suggesting that larger garage space is valued in the housing market.
# 
# **Year Built:** There’s a moderate negative correlation between Year Built and Garage Yr Blt, which could imply that newer houses might have older garages.

# # Encoding Categorical Variables

# In[18]:


df = pd.get_dummies(df, drop_first=False)
df.shape


# # Lets do some visualisations to understand our data better!
# ### 1. Plot the distribution of SalePrice.

# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, color='blue')
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()


# The plot shows the distribution of sale prices in the form of a histogram. The distribution appears to be right-skewed, with the majority of the sale prices concentrated towards the lower end of the range, while there is a long tail extending towards higher sale prices.
# - The histogram shows us a peak around 100,000 to 200,000, indicating that a significant number of properties were sold in that price range. As the sale price increases, the frequency gradually decreases, with a long tail of less frequent but higher sale prices.
# - The blue curve overlaid on the histogram represents the kernel density estimate (KDE), which provides us a smooth representation of the underlying probability density function of the sale prices.

# ### 2. Visualize how SalePrice varies with the living area.

# In[20]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df, color='blue')
plt.title('Living Area vs. Sale Price')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Sale Price')
plt.show()


# The scatter plot shows us the relationship between the living area and the sale price of properties. Each blue dot represents a single property, with its living area on the x-axis and the corresponding sale price on the y-axis. From the plot, we can observe the following:
# 
# 1. There is a generally **positive correlation** between living area and sale price. As the living area increases, the sale prices tend to increase as well, indicating that larger living areas are associated with higher sale prices.
# 2. The majority of data points are concentrated in the lower living area range, roughly between 1,000 and 3,000 square feet. This suggests that most properties in the dataset have living areas within this range.
# 3. While there is a positive correlation, the data points are spread out, indicating variability in the sale prices for a given living area. This variability could be due to other factors influencing the sale price, such as location, neighborhood, property condition, or its amenities.
# 4. There are a few data points that appear to be **outliers**, with very high sale prices relative to their living area or vice versa. These outliers may represent unique properties or even potential data errors.

# ### 3. Distribution of Sale Prices by Neighborhood.

# In[21]:


neighborhood_cols = [col for col in df.columns if col.startswith('Neighborhood')]
neighborhood_prices = df[neighborhood_cols + ['SalePrice']].copy()
neighborhood_prices_melted = neighborhood_prices.melt(id_vars=['SalePrice'], value_vars=neighborhood_cols, var_name='Neighborhood', value_name='Neighborhood_Flag')

neighborhood_prices_melted = neighborhood_prices_melted[neighborhood_prices_melted['Neighborhood_Flag'] == 1]

neighborhood_prices_melted.drop(columns=['Neighborhood_Flag'], inplace=True)


# In[22]:


plt.figure(figsize=(20, 10))
for i, neighborhood in enumerate(neighborhood_cols, start=1):
    plt.subplot(5, 6, i)
    sns.histplot(data=neighborhood_prices_melted[neighborhood_prices_melted['Neighborhood'] == neighborhood], x='SalePrice', kde=True)
    plt.title(neighborhood)
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# # Splitting the data into train, test & validation sets

# In[23]:


X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state = 42)


# We are done with pre-processing and splitting of the data. Now we can proceed with Model and Evaluation. The models used for this dataset is as follows:
#  - Linear Regression
#  - Decision tree
#  - Gradient Boosting
#  - AdaBoost
#  - Support Vector Machines

# # Function to compute metrics

# In[24]:


def metrics_compute(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


# # DataFrame to store metrics

# In[25]:


metrics_columns = ['R2', 'RMSE', 'MAE']
metrics_df = pd.DataFrame(columns=['Model', 'Dataset'] + metrics_columns)


# # Linear Regression

# In[26]:


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_train_pred_lr = linear_reg.predict(X_train)
y_test_pred_lr = linear_reg.predict(X_test)
y_val_pred_lr = linear_reg.predict(X_val)
metrics_df = metrics_df.append(pd.DataFrame([
    ['Linear Regression', 'Training'] + list(metrics_compute(y_train, y_train_pred_lr)),
    ['Linear Regression', 'Testing'] + list(metrics_compute(y_test, y_test_pred_lr)),
    ['Linear Regression', 'Validation'] + list(metrics_compute(y_val, y_val_pred_lr)),
], columns=metrics_df.columns))


# # Decision Tree

# In[27]:


dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)
y_val_pred_dt = dt_model.predict(X_val)
metrics_df = metrics_df.append(pd.DataFrame([
    ['Decision Tree', 'Training'] + list(metrics_compute(y_train, y_train_pred_dt)),
    ['Decision Tree', 'Testing'] + list(metrics_compute(y_test, y_test_pred_dt)),
    ['Decision Tree', 'Validation'] + list(metrics_compute(y_val, y_val_pred_dt)),
], columns=metrics_df.columns))


# # Gradient Boosting

# In[28]:


gbm = GradientBoostingRegressor(random_state = 42)
gbm.fit(X_train, y_train)
y_train_pred_gb = gbm.predict(X_train)
y_test_pred_gb = gbm.predict(X_test)
y_val_pred_gb = gbm.predict(X_val)
metrics_df = metrics_df.append(pd.DataFrame([
    ['Gradient Boosting', 'Training'] + list(metrics_compute(y_train, y_train_pred_gb)),
    ['Gradient Boosting', 'Testing'] + list(metrics_compute(y_test, y_test_pred_gb)),
    ['Gradient Boosting', 'Validation'] + list(metrics_compute(y_val, y_val_pred_gb)),
], columns=metrics_df.columns))


# # AdaBoost

# In[29]:


ada_regr = AdaBoostRegressor(random_state = 42, n_estimators=500)
ada_regr.fit(X_train, y_train)
y_train_pred_AB = ada_regr.predict(X_train)
y_test_pred_AB = ada_regr.predict(X_test)
y_val_pred_AB = ada_regr.predict(X_val)
metrics_df = metrics_df.append(pd.DataFrame([
    [' AdaBoost', 'Training'] + list(metrics_compute(y_train, y_train_pred_AB)),
    ['AdaBoost', 'Testing'] + list(metrics_compute(y_test, y_test_pred_AB)),
    ['AdaBoost', 'Validation'] + list(metrics_compute(y_val, y_val_pred_AB)),
], columns=metrics_df.columns))


# # SVM

# In[30]:


svm_model = SVR()
svm_model.fit(X_train, y_train)
y_train_pred_svm = svm_model.predict(X_train)
y_test_pred_svm = svm_model.predict(X_test)
y_val_pred_svm = svm_model.predict(X_val)
metrics_df = metrics_df.append(pd.DataFrame([
    ['SVM', 'Training'] + list(metrics_compute(y_train, y_train_pred_svm)),
    ['SVM', 'Testing'] + list(metrics_compute(y_test, y_test_pred_svm)),
    ['SVM', 'Validation'] + list(metrics_compute(y_val, y_val_pred_svm)),
], columns=metrics_df.columns))


# In[31]:


metrics_df


# Here’s the summary of the model findings:
# 
# **Linear Regression**
# 
# Shows good performance on the training dataset with an R2 of 0.925860, but a drop in performance on the testing dataset with an R2 of 0.863908 indicating potential overfitting.
# 
# **Decision Tree**
# 
# Perfect on the training dataset R2 of 1.000000 but less effective on the testing dataset R2 of 0.820335
# 
# **Gradient Boosting**
# 
# Strong performance on both training R2 of 0.963355 and testing datasets R2 of 0.908683, indicating robustness.
# 
# **AdaBoost**
# 
# Moderate performance with an R2 of 0.837092 on the training set and 0.808471 on the testing set, indicating stability between the sets.
# 
# **SVM (Support Vector Machine)**
# 
# Poor performance with negative R2 values on both training and testing datasets, suggesting it may not be suitable for this particular problem.

# # Feature Engineering And Hypermeter Tuning

# In[32]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


# # Gradient Boosting with GridSearchCV

# In[33]:


param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 2, 4]
}


# In[34]:


gbm = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_gbm = grid_search.best_estimator_
y_train_pred_gb = best_gbm.predict(X_train_scaled)
y_test_pred_gb = best_gbm.predict(X_test_scaled)
y_val_pred_gb = best_gbm.predict(X_val_scaled)
metrics_df = pd.concat([metrics_df, pd.DataFrame([
    ['Gradient Boosting with GridSearch', 'Training'] + list(metrics_compute(y_train, y_train_pred_gb)),
    ['Gradient Boosting with GridSearch', 'Testing'] + list(metrics_compute(y_test, y_test_pred_gb)),
    ['Gradient Boosting with GridSearch', 'Validation'] + list(metrics_compute(y_val, y_val_pred_gb)),
], columns=metrics_df.columns)])


# In[35]:


metrics_df


# Both the default and tuned Gradient Boosting models perform well, with the tuned version showing improved performance across all datasets. This indicates the effectiveness of hyperparameter tuning.

# Given the hyperparameter tuning results, the best performing model is the **Gradient Boosting** with **GridSearch**. 
# 
# ### Lets perform Cross Validation with Gradient Boosting!

# In[38]:


cv_results = cross_val_score(best_gbm, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_pred = cross_val_predict(best_gbm, X_train_scaled, y_train, cv=5)

cv_r2 = r2_score(y_train, cv_pred)
cv_rmse = mean_squared_error(y_train, cv_pred, squared=False)
cv_mae = mean_absolute_error(y_train, cv_pred)

metrics_df = metrics_df.append(pd.DataFrame([
    ['Gradient Boosting with CV', 'Cross-Validation'] + [cv_r2, cv_rmse, cv_mae]
], columns=metrics_df.columns))


# In[39]:


metrics_df


# The cross validation results confirm that the Gradient Boosting model tuned with GridSearchCV performs well. 

# # Predicting House Sale Prices with the Best Model

# In[40]:


best_gbm.fit(X_train_scaled, y_train)


# In[41]:


metrics_columns = ['R2', 'RMSE', 'MAE']
metrics_bestFit = pd.DataFrame(columns=['Model', 'Dataset'] + metrics_columns)


# In[42]:


y_test_pred_best_gbm = best_gbm.predict(X_test_scaled)
y_val_pred_best_gbm = best_gbm.predict(X_val_scaled)
metrics_bestFit = pd.concat([metrics_bestFit, pd.DataFrame([
    ['Gradient Boosting with GridSearch and best fit', 'Testing'] + list(metrics_compute(y_test, y_test_pred_best_gbm)),
    ['Gradient Boosting with GridSearch and best fit', 'Validation'] + list(metrics_compute(y_val, y_val_pred_best_gbm)),
], columns=metrics_bestFit.columns)])


# ### Lets visualize the sale price predictions made by the best Gradient Boosting model
# 
# ### Test Data

# In[45]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_best_gbm, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Sale Prices on Test Dataset')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.show()


# Here, The scatter plot compares the actual sale prices and the predicted sale prices made by the best Gradient Boosting model on the test dataset.
# - The blue dots represent the data points, with the x-axis showing the actual sale prices and the y-axis showing the predicted sale prices. The red dashed line represents the perfect prediction line, where the predicted sale prices would be equal to the actual sale prices.
# 
# From the plot we can observe that most of the data points are clustered around the perfect prediction line, indicating that the model is performing reasonably well in predicting the sale prices. The plot also shows that as the actual sale prices increase, the predicted sale prices tend to be slightly overestimated by the model, as indicated by the data points being above the perfect prediction line for higher sale prices.

# ### Validation Data

# In[46]:


plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred_best_gbm, color='blue', alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
plt.title('Actual vs Predicted Sale Prices on Validation Dataset')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.show()


# This plot compares the actual sale prices and the predicted sale prices made by the best Gradient Boosting model on the validation dataset.
# 
# From this we can observe that the data points are generally clustered around the perfect prediction line, indicating that the model is performing well aswell. Compared to the test dataset plot, the data points in this plot appear to be more evenly distributed around the perfect prediction line, suggesting that the model's performance on the validation dataset may be `slightly better` than its performance on the test dataset. As with the test dataset plot, we can see that for higher actual sale prices, the model tends to overestimate the predicted sale prices, with more data points falling above the perfect prediction line in that range.

# In[47]:


metrics_bestFit


# # Conclusion
# 
# The Gradient Boosting model, optimized using GridSearchCV, has demonstrated strong performance in predicting house sale prices. The evaluation metrics on the test and validation datasets are as follows:
# 
# **Test Dataset:**
# - R² Score: 0.911
# - RMSE: 25,314.97
# - MAE: 14,136.20
# 
# **Validation Dataset:**
# - R² Score: 0.940
# - RMSE: 20,214.73
# - MAE: 13,529.91
# 
# These metrics indicate that the model is effective at predicting house prices, with high R² scores showing that a large proportion of the variance in house prices is explained by the model. The relatively low values of RMSE and MAE suggest that the model's predictions are close to the actual values, both in terms of average and absolute errors.
# 
# To further enhance the model's performance, several strategies could be implemented. Advanced feature engineering, including interaction and polynomial features, can capture more complex relationships. More sophisticated hyperparameter tuning methods, like RandomizedSearchCV or Bayesian optimization, and the exploration of ensemble methods can also yield improvements. Additional data collection, data augmentation, and outlier treatment will strengthen the model's robustness. Trying different models, implementing regularization techniques, and conducting thorough error analysis will further refine the model. Utilizing techniques like K-Fold Cross-Validation, stacking, and AutoML can ensure consistent and optimal performance which can be implemented when we have further time to proceed with to increase the performance of the prediction.
# 
# While the Gradient Boosting model already shows impressive predictive capabilities, there are multiple avenues for further improvement. By exploring these additional strategies, we can enhance the accuracy and reliability of house sale price predictions even further.
