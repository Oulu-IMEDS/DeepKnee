import {Component} from "react";
import React from "react";


class Footer extends Component {
    render() {
        return (
            <nav className="navbar fixed-bottom navbar-light bg-light justify-content-center">
                &copy; 2018-2019 Aleksei Tiulpin & Egor Panfilov.
                &nbsp; <b>|</b>&nbsp; Saarakkala's group &nbsp; <b>|</b> &nbsp; University of Oulu. This software is not cleared for diagnostic purposes.
            </nav>
        );
    }
}

export default Footer;