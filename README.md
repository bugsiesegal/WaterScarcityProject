# Water Scarcity AI 💧

## Overview
A machine learning model developed during a hackathon to predict water scarcity in different regions using satellite data from NASA's Earth Data platform.

## Features
- Downloads and processes Earth observation data
- Predicts water scarcity risk for specified regions
- Uses both PyTorch and Keras for model implementation
- Visualizes results using matplotlib

## Project Structure
```
water-scarcity-ai/
├── ai_model.py         # Model architecture and training code
├── datadownloader.py   # Earth Data download functionality
├── dataprocessing.py   # Data preprocessing pipeline
├── model.pt           # Trained PyTorch model weights
└── test.py            # Inference and testing script
```

## Prerequisites
### Dependencies
```bash
pip install torch torchvision keras netCDF4 matplotlib earthdata
```

### Earth Data Credentials
1. Create an account at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Set up your credentials:
   ```python
   # In your environment or .env file
   EARTHDATA_USERNAME=your_username
   EARTHDATA_PASSWORD=your_password
   ```

## Usage
1. Download the required data:
   ```bash
   python datadownloader.py
   ```

2. Run predictions:
   ```bash
   python test.py
   ```

## Data Sources
- Uses NASA Earth Data satellite observations
- Processes netCDF4 format files

## Model Details
- Hybrid architecture using both PyTorch and Keras
- Input: Processed satellite data
- Output: Water scarcity risk prediction

## Visualization
The system generates plots showing:
- Predicted water scarcity regions

## Performance
- Developed and tested during hackathon conditions
- Capable of processing regional data

## Limitations
- Requires Earth Data credentials
- Processing large regions may be time-intensive
- Limited to available satellite data coverage

## Future Improvements
- Add more data sources
- Improve prediction accuracy
- Optimize processing pipeline
- Add real-time monitoring capabilities
- Expand geographical coverage

## Contributing
Feel free to fork and submit pull requests. Areas that would benefit from contribution:
- Additional data sources
- Model improvements
- Processing optimization
- Documentation

## Acknowledgments
- NASA Earth Data for satellite data access
- Hackathon organizers and participants
- (Add other acknowledgments)

---
*Created during a hackathon in June 2022*
