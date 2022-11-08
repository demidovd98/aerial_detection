console.log(__dirname);
const fs = require('fs');
fs.readFile('iSAID/test_info.json', function(error, data) {
    if (error) {
        console.log(error);
        return;
    }

    var obj = JSON.parse(data);
    console.log(obj.images[0])
    console.log(obj.images[0].file_name)
    console.log(obj.images[0].id)
   
    for(var p in obj.images) {
        fs.rename('iSAID/images/test/images/' + obj.images[p].file_name, 'iSAID/images/test/images2/' + obj.images[p].id + '.png', function(err) {
            if ( err ) console.log('ERROR: ' + err);
        });
    }


});
