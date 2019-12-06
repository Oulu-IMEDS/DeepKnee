import React, {Component} from "react";

class FileUploader extends Component {
    constructor(props) {
        super(props);
        this.state = {file_name: ''};

        this.handleFileChosen = this.handleFileChosen.bind(this);
        this.handleFileSubmitted = this.handleFileSubmitted.bind(this);
    }

    extractFilename = function (fullname) {
        return fullname.split('\\').pop().split('/').pop();
    };

    handleFileChosen(event) {
        const file_name = this.extractFilename(event.target.value);
        this.setState({file_name: file_name});
    }

    handleFileSubmitted(event) {
        let file_input = document.querySelector('input[id=inputGroupFile]');
        let file = file_input.files[0];

        let reader = new FileReader();

        reader.onloadend = () => {
            let blob = {
                file_name: this.state.file_name,
                file_blob: reader.result,
            };
            console.log('File loaded');
            this.props.onFileSubmission(blob);

            file_input.value = "";
            this.setState({file_name: ''});
        };
        // Read file as base64 string
        reader.readAsDataURL(file);
    }

    render() {
        return (
            <div className="input-group my-3">
                <div className="input-group-prepend">
                    <button className="btn btn-outline-secondary" type="button"
                            onClick={this.handleFileSubmitted}>
                        Submit DICOM
                    </button>
                </div>
                <div className="custom-file">
                    <input type="file" className="custom-file-input" id="inputGroupFile"
                           onChange={this.handleFileChosen} />
                    <label className="custom-file-label" htmlFor="inputGroupFile">
                        {this.state.file_name ? this.state.file_name : <i>Choose file</i>}
                    </label>
                </div>
            </div>
        )};
}

export default FileUploader;
