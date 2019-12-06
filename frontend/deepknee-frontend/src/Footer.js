import {Component} from "react";
import React from "react";


class Footer extends Component {
    render() {
        return (
            <nav className="navbar fixed-bottom navbar-light bg-light justify-content-center">
                (c) Aleksei Tiulpin, Egor Panfilov and Simo Saarakkala
                &nbsp; <b>|</b> &nbsp; University of Oulu, 2018-2019. This software is not cleared for diagnostic purposes.
            </nav>
        );
    }
}

export default Footer;