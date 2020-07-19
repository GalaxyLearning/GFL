pragma solidity >=0.4.22 <= 0.6.11;


contract Storage {


    struct ModelParameter{

        address owner;
        string modelParameter;
    }


    uint fedStep;

    address[] activateClients;

    address[] selectedClients;

    // fed step -> model parameters
    mapping(uint => ModelParameter[]) public modelParameters;

    // fed step -> selected client address
    mapping(uint => address) selectedAddress;

    // fed step -> aggregated model parameters
    mapping(uint => ModelParameter) public aggregatedParameters;

    string final_aggregated_model_parameters public;

    function __addressValidation(address sender) private view returns(bool) {

        for(uint i=0; i<activateClients.length; i++){
            if(sender == activateClients[i]){
                return true;
            }
        }

        return false;
    }

    function setFedStep(uint _fedStep) public {
        fedStep = _fedStep;
    }

    function getFedStep() public view returns (uint) {
        return fedStep;
    }

    function updateActivateClients(address _clientAddress, uint8 op) public {

        uint sym = 0;
        bool isHave = false;
        for(uint i=0; i<activateClients.length; i++){
            if(_clientAddress == activateClients[i]){
                isHave = true;
                sym = i;
                break;
            }
        }
        if(!isHave && op == 0){
            activateClients.push(_clientAddress);
        }else if(isHave  && op == 1){
            activateClients[sym] = activateClients[activateClients.length - 1];
            delete activateClients[activateClients.length - 1];
            activateClients.length-- ;
        }

    }

    function uploadModelParametersIpfsHash(uint _fedStep, address _clientAddress, string memory _modelParsIpfsHash) public {

        ModelParameter memory modelParameter = ModelParameter(_clientAddress, _modelParsIpfsHash);

        modelParameters[_fedStep].push(modelParameter);
    }

    // function downloadModelParametersIpfsHash(uint _fedStep, address _clientAddress) public view returns(ModelParameter) {

    //     return modelParameters[_fedStep];

    // }
    function uploadAggreagtedParametersIpfsHash(uint _fedStep, address _modelClientAddress, string memory _aggregatedModelParsIpfsHash) public {

        ModelParameter memory modelParameter = ModelParameter(_modelClientAddress, _aggregatedModelParsIpfsHash);

        aggregatedParameters[_fedStep] = modelParameter;

    }

    function uploadFinalAggregatedParametersIpfsHash(string memory _finalAggregatedIpfsHash) public {

        final_aggregated_model_parameters = _finalAggregatedIpfsHash;
    }






}