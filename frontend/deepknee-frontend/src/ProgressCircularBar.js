import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import CircularProgress from '@material-ui/core/CircularProgress';
import Paper from '@material-ui/core/Paper';
import green from '@material-ui/core/colors/green';
import blue from '@material-ui/core/colors/lightBlue';
import purple from '@material-ui/core/colors/purple';


const styles = theme => ({
  progress: {
    margin: theme.spacing.unit * 2,
  },
});

const colors = {
    "blue": blue[500],
    "green": green[500],
    "purple": purple[500]
};

function ProgressCircularBar(props) {
  const { classes } = props;

  return (
    <div className="col" style={{height: "400px"}} align="center">
        <div className="row" style={{height: "50%", justifyContent:'center', alignItems:'center'}}>
            <CircularProgress className={classes.progress} size={100}
                              style={{color: colors[props.alternative]}} />
        </div>
        <div className="col" style={{height: "50%", width: "50%", justifyContent:'center', alignItems:'center'}}>
            <Paper elevation={0}>{props.text}</Paper>
        </div>
    </div>
  );
}

ProgressCircularBar.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(ProgressCircularBar);