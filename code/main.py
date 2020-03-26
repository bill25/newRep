from settings import settings
from object_factory import object_factory
from mappers import portfolios_allocation_mapper


import pandas as pd
from pymongo import MongoClient


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]


def read_mongo(db, collection,instrument , host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """


    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    # Make a query to the specific DB and Collection
    query = {"instrument": instrument}
    cursor = db[collection].find(query)
    for x in cursor:
         df= pd.DataFrame(x.get('candles'))[['time','mid']]


    df1=df['mid'].apply(pd.Series)
    df=pd.concat([df[:], df1[:]], axis=1)
    df=df[['time','c']]
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return(df.rename(columns={'c':instrument,'time':'Date'}))







def generate_optimum_portfolio():


    #instantiate the objects with the settings
    obj_factory = object_factory(settings)
    companie_extractor = obj_factory.get_companies_extractor()
    cp = obj_factory.get_charts_plotter()
    mcs = obj_factory.get_portfolio_generator()
    fr = obj_factory.get_file_repository()
    metricCalculator = obj_factory.get_metrics_calculator()
    print('1. Get companies')
    companies = companie_extractor.get_companies_list()
    print("hello companies  :",companies)
    price_extractor = obj_factory.get_price_extractor(companies)



    print('2. Get company stock prices')

    end_date = settings.get_end_date()
    start_date = settings.get_start_date(end_date)
    closing_prices = price_extractor.get_prices(settings.PriceEvent, start_date, end_date)
    print(closing_prices.dtypes)


    closing_prices
    #plot stock prices & save data to a file
    cp.plot_prices(closing_prices)
    fr.save_to_file(closing_prices, 'StockPrices')

    print('3. Calculate Daily Returns')
    returns = settings.DailyAssetsReturnsFunction(closing_prices, settings.ReturnType)

    #plot stock prices & save data to a file
    cp.plot_returns(returns)
    fr.save_to_file(returns, 'Returns')

    print('4. Calculate Expected Mean Return & Covariance')
    expected_returns = settings.AssetsExpectedReturnsFunction(returns)


    covariance = settings.AssetsCovarianceFunction(returns)

    #Plot & Save covariance to file
    cp.plot_correlation_matrix(returns)
    fr.save_to_file(covariance, 'Covariances')

    print('5. Use Monte Carlo Simulation')
    #Generate portfolios with allocations
    portfolios_allocations_df = mcs.generate_portfolios(expected_returns, covariance, settings.RiskFreeRate)

    portfolio_risk_return_ratio_df = portfolios_allocation_mapper.map_to_risk_return_ratios(portfolios_allocations_df)

    #Plot portfolios, print max sharpe portfolio & save data

    cp.plot_portfolios(portfolio_risk_return_ratio_df)

    max_sharpe_portfolio = metricCalculator.get_max_sharpe_ratio(portfolio_risk_return_ratio_df)['Portfolio']
    max_shape_ratio_allocations = portfolios_allocations_df[[ 'Symbol', max_sharpe_portfolio]]
    print("optimum portfolio ",max_shape_ratio_allocations)


    fr.save_to_file(portfolios_allocations_df, 'MonteCarloPortfolios')
    fr.save_to_file(portfolio_risk_return_ratio_df, 'MonteCarloPortfolioRatios')

    print('6. Use an optimiser')
    #Generate portfolios
    targets = settings.get_my_targets()
    print('targets',   targets)
    optimiser = obj_factory.get_optimiser(targets, len(expected_returns.index))

    portfolios_allocations_df = optimiser.generate_portfolios(expected_returns, covariance, settings.RiskFreeRate)
    portfolio_risk_return_ratio_df = portfolios_allocation_mapper.map_to_risk_return_ratios(portfolios_allocations_df)
    #plot efficient frontiers
    cp.plot_efficient_frontier(portfolio_risk_return_ratio_df)
    cp.show_plots()

    #save data
    print('7. Saving Data')
    fr.save_to_file(portfolios_allocations_df, 'OptimisationPortfolios')
    fr.close()

generate_optimum_portfolio()
