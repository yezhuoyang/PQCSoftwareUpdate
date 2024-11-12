


struct Keypair{
    publickey: u32,
    secretekey:u32
}


impl Keypair{
    

    fn new(publickey: u32, secretekey: u32) -> Keypair{
        Keypair{publickey,secretekey}
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
    let mut keypair=Keypair::new(31,32);
    keypair.printkey();
    keypair.setkeys(3241,1231);
    keypair.printkey();
}