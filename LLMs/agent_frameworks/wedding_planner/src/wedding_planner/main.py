#!/usr/bin/env python
import sys
from wedding_planner.crew import DestinationWeddingPlannerCrew
import os

os.environ["OTEL_SDK_DISABLED"] = "true"


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'wedding_style': 'bengali',
        'wedding_budget': '1 crore',
        'preferred_country': 'Bali',
        'groom_culture': 'Kannada',
        'bride_culture': 'Bengali',
        'csv': 'country_venue_price.csv',
    }
    DestinationWeddingPlannerCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'wedding_style': 'bengali',
        'wedding_budget': '1 crore',
        'preferred_country': 'Bali',
        'groom_culture': 'Kannada',
        'bride_culture': 'Bengali',
        'csv': 'country_venue_price.csv',
    }
    try:
        DestinationWeddingPlannerCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
