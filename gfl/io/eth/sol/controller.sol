pragma solidity >=0.4.22 <0.6.0;

contract Controller {


    mapping(string => address) jobMappingStorageContract;

    mapping(string => uint8) jobStatus;

    function setJobMappingStorageContract(string memory jobId, address storageContractAddress) public {

        jobMappingStorageContract[jobId] = storageContractAddress;
    }

    function getJobMappingStorageContract(string memory jobId) public view returns (address){
        return jobMappingStorageContract[jobId];
    }

    function setJobStatus(string memory jobId, uint8 status) public {
        jobStatus[jobId] = status;
    }

    function getJobStatus(string memory jobId) public view returns (uint8) {
        return jobStatus[jobId];
    }


}