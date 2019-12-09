import React, {Component} from "react";
import openSocket from 'socket.io-client';

class SIOClient extends Component {
    constructor(props) {
        super(props);
        const port = process.env.REACT_APP_BROKER_PORT || "4001";
        const host = process.env.REACT_APP_BROKER_ADDR || "http://127.0.0.1";

        this.state = {
            endpoint:  `${host}:${port}/`
        };
        this.socket = openSocket(
            this.state.endpoint,
            {path: `/socket.io`,
             timeout: 120000,
             pingTimeout: 120000,
             pingInterval: 120000,
            }
        );
    }

    componentDidMount() {
        this.socket.on("dicom_received", data => this.props.onDicomReceived(data));
        this.socket.on("dicom_processed", data => this.props.onDicomProcessed(data));
        this.socket.on("connect", () => this.props.onServerConnected());
        this.socket.on("disconnect", () => this.props.onServerDisconnected());
    }

    componentDidUpdate(prevProps, prevState) {
        // Input file updated
        if (this.props.file_blob !== prevProps.file_blob) {
            let blob = {
                file_name: this.props.file_name,
                file_blob: this.props.file_blob,
            };

            this.socket.emit('dicom_submission', blob);
            this.props.onDicomSent();
        }
    }

    render() {
        return (
            this.props.connected ? null :
                <div className="alert alert-danger" role="alert">
                    Server is not connected
                </div>
        );
    }
}

export default SIOClient;
