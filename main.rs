use std::io;
use rand::Rng;




struct keypair{
    publickey: u32,
    secretekey:u32
}


impl keypair{
    

    fn new(publickey: u32, secretekey: u32) -> keypair{
        keypair{publickey,secretekey}
    }

    fn printkey(&self){
        println!("The keypair is {} , {}",self.publickey,self.secretekey);
    }

    fn setkeys(&mut self,publickey: u32, secretekey: u32){
        self.publickey=publickey;
        self.secretekey=secretekey;
    }


}


fn main(){
    let mut keypair=keypair::new(31,32);
    keypair.printkey();
    keypair.setkeys(3241,1231);
    keypair.printkey();
}