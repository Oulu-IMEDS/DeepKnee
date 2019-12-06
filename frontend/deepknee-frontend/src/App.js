import React, {Component} from 'react';
import SIOClient from './SIOClient';
import FileUploader from './FileUploader';
import ProgressCircularBar from './ProgressCircularBar';
import Footer from './Footer';

class App extends Component {
    constructor(props) {
        super(props);
        this.state = {
            server_connected: false,
            server_status: null,
            server_response: null,

            file_name: null,
            file_blob: null,

            image_1st_raw: null,
            image_2nd_raw: null,
            image_1st_heatmap: null,
            image_2nd_heatmap: null,
            special_1st: null,
            special_2nd: null,
        };

        this.handleFileSubmission = this.handleFileSubmission.bind(this);

        this.handleDicomSent = this.handleDicomSent.bind(this);
        this.handleDicomReceived = this.handleDicomReceived.bind(this);
        this.handleDicomProcessed = this.handleDicomProcessed.bind(this);
        this.handleServerConnected = this.handleServerConnected.bind(this);
        this.handleServerDisconnected = this.handleServerDisconnected.bind(this);
    }

    handleFileSubmission(data) {
        this.setState({
            is_waiting_response: true,

            file_name: data.file_name,
            file_blob: data.file_blob
        });
    }

    handleDicomSent(data) {
        this.setState({
            server_status: "dicom_sent",
        });
        console.log("Sent");
    }

    handleDicomReceived(data) {
        this.setState({
            server_status: "dicom_received",
        });
        console.log("Received");
    }

    handleDicomProcessed(data) {
        this.setState({
            server_status: "dicom_processed",
            server_response: data.server_response,

            image_1st_raw: data.image_1st_raw,
            image_2nd_raw: data.image_2nd_raw,
            image_1st_heatmap: data.image_1st_heatmap,
            image_2nd_heatmap: data.image_2nd_heatmap,
            special_1st: data.special_1st,
            special_2nd: data.special_2nd,
        });
        console.log("Processed");
    }

    handleServerConnected() {
        this.setState({
            server_connected: true,
            server_status: 'standby',
        });
    }

    handleServerDisconnected() {
        this.setState({
            server_connected: false,
            server_status: 'standby',
        });
    }

    render() {
        const state = this.state;
        return (
            <div className="col">
                <SIOClient
                    file_name={state.file_name}
                    file_blob={state.file_blob}
                    connected={state.server_connected}
                    response={state.server_response}
                    onDicomSent={this.handleDicomSent}
                    onDicomReceived={this.handleDicomReceived}
                    onDicomProcessed={this.handleDicomProcessed}
                    onServerConnected={this.handleServerConnected}
                    onServerDisconnected={this.handleServerDisconnected}
                />

                <FileUploader
                    onFileSubmission={this.handleFileSubmission}
                />

                <hr />

                {state.server_status === "dicom_sent" &&
                <ProgressCircularBar alternative={"blue"} text={"Sending the file..."} />}

                {state.server_status === "dicom_received" &&
                <ProgressCircularBar alternative={"green"} text={"Analyzing the image..."} />}

                {state.server_status === "dicom_processed" &&
                <div className="container" style={{visibility: state.image_1st_raw == null ? "hidden" : ""}}>
                    <div className="row" style={{height: "20px"}}>
                        <div className="col-sm">
                            <p className="text-center font-weight-bold">Left knee</p>
                        </div>
                        <div className="col-sm">
                            <p className="text-center font-weight-bold">Right knee</p>
                        </div>
                    </div>
                    <div className="row" style={{height: "400px"}}>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.image_1st_raw} className="img-fluid" alt=""/>
                        </div>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.image_1st_heatmap} className="img-fluid" alt=""/>
                        </div>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.image_2nd_raw} className="img-fluid" alt=""/>
                        </div>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.image_2nd_heatmap} className="img-fluid" alt=""/>
                        </div>
                    </div>
                    <div className="row" style={{height: "100px"}}>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.special_1st} className="img-fluid" alt=""/>
                        </div>
                        <div className="col-sm text-center align-self-center">
                            <img src={state.special_2nd} className="img-fluid" alt=""/>
                        </div>
                    </div>
                </div>
                }

                <Footer/>
            </div>
        );
    }
}

export default App;
